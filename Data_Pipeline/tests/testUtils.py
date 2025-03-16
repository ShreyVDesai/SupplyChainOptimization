import io
import os
import json
import unittest
from unittest.mock import MagicMock, call, patch, Mock

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
from scripts.utils import (
    load_bucket_data,
    send_email,
    setup_gcp_credentials,
    upload_to_gcs,
    send_anomaly_alert,
    collect_validation_errors,
    delete_blob_from_bucket,
    list_bucket_blobs
)


class TestUtils(unittest.TestCase):

    @patch("scripts.utils.logger")
    @patch("scripts.utils.pl.read_excel")
    @patch("scripts.utils.storage.Client")
    def test_load_bucket_data_success(
        self, mock_storage_client, mock_read_excel, mock_logger
    ):
        # Setup
        dummy_bytes = b"dummy excel bytes"
        expected_df = pl.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Unit Price": [100.5, 200.75],
                "Quantity": [10, 20],
                "Transaction ID": ["T123", "T456"],
                "Store Location": ["NY", "CA"],
                "Product Name": ["milk", "coffee"],
                "Producer ID": ["P001", "P002"],
            }
        )

        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        mock_read_excel.return_value = expected_df
        bucket_name = "test-bucket"
        file_name = "test-file.xlsx"

        # Test
        result = load_bucket_data(bucket_name, file_name)

        # Assert
        mock_storage_client.assert_called_once()
        mock_client_instance.get_bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(file_name)
        mock_blob.download_as_string.assert_called_once()

        args, _ = mock_read_excel.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        self.assertEqual(args[0].getvalue(), dummy_bytes)

        expected_log = f"'test-file.xlsx' from bucket 'test-bucket' successfully read as Excel into DataFrame."
        mock_logger.info.assert_any_call(expected_log)

        assert_frame_equal(result, expected_df)

    @patch("scripts.utils.logger")
    @patch("scripts.utils.storage.Client")
    def test_load_bucket_data_failure_exception_handling(
        self, mock_storage_client, mock_logger
    ):
        """
        Test that if blob.download_as_string() raises an exception,
        the function logs an error and re-raises the exception.
        """
        error_message = "Simulated download error"

        # Set up a mock blob that raises an exception on download.
        mock_blob = MagicMock()
        mock_blob.download_as_string.side_effect = Exception(error_message)

        # Set up a mock bucket that returns our mock blob.
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        # Set up a mock client instance whose get_bucket returns the mock
        # bucket.
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        bucket_name = "test-bucket"
        file_name = "test-file.xlsx"

        with self.assertRaises(Exception) as context:
            load_bucket_data(bucket_name, file_name)

        # Verify the exception message contains our simulated error.
        self.assertIn(error_message, str(context.exception))

        # Verify that logger.error was called with a message that includes the
        # error.
        mock_logger.error.assert_called_once()
        error_log_msg = mock_logger.error.call_args[0][0]
        self.assertIn(error_message, error_log_msg)

    @patch("scripts.utils.logger")
    @patch("scripts.utils.pl.read_excel")
    @patch("scripts.utils.storage.Client")
    def test_empty_dataframe_raises_value_error_load(
        self, mock_storage_client, mock_read_excel, mock_logger
    ):
        # Setup
        empty_df = pl.DataFrame([])
        mock_read_excel.return_value = empty_df

        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = b"dummy_bytes"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        bucket_name = "dummy-bucket"
        file_name = "dummy.xlsx"

        # Test
        with self.assertRaises(ValueError) as context:
            load_bucket_data(bucket_name, file_name)

        # Assert
        self.assertIn("is empty", str(context.exception))
        expected_substring = "DataFrame loaded from bucket 'dummy-bucket', file 'dummy.xlsx' is empty."
        error_messages = [
            args[0][0] for args in mock_logger.error.call_args_list
        ]
        self.assertTrue(
            any(expected_substring in message for message in error_messages),
            f"None of the error messages contained '{expected_substring}'.",
        )

    # Unit tests for send_email

    # Test case where send_email executes successfully without attachment.
    @patch("scripts.utils.logger")
    @patch("scripts.utils.smtplib.SMTP")
    def test_send_email_success_without_attachment(
        self, mock_smtp, mock_logger
    ):
        # Setup
        smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = smtp_instance

        emailid = "recipient@example.com"
        body = "This is a test email."
        subject = "Test Email"

        # Test
        send_email(emailid, body, subject=subject)

        # Assert
        mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once_with(
            "talksick530@gmail.com", "celm dfaq qllh ymjv"
        )
        smtp_instance.send_message.assert_called_once()
        mock_logger.info.assert_any_call(
            f"Email sent successfully to: {emailid}"
        )

    # Test case where send_email executes successfully with attachment.
    @patch("scripts.utils.logger")
    @patch("scripts.utils.smtplib.SMTP")
    def test_send_email_success_with_attachment(self, mock_smtp, mock_logger):
        # Setup
        smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = smtp_instance

        emailid = "recipient@example.com"
        body = "This is a test email with attachment."
        subject = "Test Email with Attachment"

        attachment_df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Unit Price": [100.5, 200.75],
                "Quantity": [10, 20],
                "Transaction ID": ["T123", "T456"],
                "Store Location": ["NY", "CA"],
                "Product Name": ["milk", "coffee"],
                "Producer ID": ["P001", "P002"],
            }
        )

        # Test
        send_email(emailid, body, subject=subject, attachment=attachment_df)

        # Assert
        smtp_instance.send_message.assert_called_once()
        sent_msg = smtp_instance.send_message.call_args[0][0]
        self.assertTrue(
            sent_msg.is_multipart(),
            "EmailMessage should be multipart when an attachment is added.",
        )
        attachments = [part for part in sent_msg.iter_attachments()]
        self.assertEqual(
            len(attachments), 1, "There should be exactly one attachment."
        )
        attachment_part = attachments[0]
        self.assertEqual(attachment_part.get_filename(), "anomalies.csv")
        csv_content = attachment_part.get_content()
        self.assertIn("Date", csv_content)
        self.assertIn("Unit Price", csv_content)
        self.assertIn("Quantity", csv_content)
        self.assertIn("Transaction ID", csv_content)
        self.assertIn("Store Location", csv_content)
        self.assertIn("Product Name", csv_content)
        self.assertIn("Producer ID", csv_content)

    # Test case where exception handling done.
    @patch("scripts.utils.logger")
    @patch("scripts.utils.smtplib.SMTP")
    def test_send_email_exception(self, mock_smtp, mock_logger):
        # Setup
        smtp_instance = MagicMock()
        smtp_instance.send_message.side_effect = Exception("SMTP error")
        mock_smtp.return_value.__enter__.return_value = smtp_instance

        emailid = "recipient@example.com"
        body = "Test email"
        subject = "Test Exception Email"

        # Test
        with self.assertRaises(Exception) as context:
            send_email(emailid, body, subject=subject)

        # Assert
        self.assertIn("SMTP error", str(context.exception))
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn("Failed to send email", error_message)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.pl.read_csv")
    @patch("scripts.utils.logger")
    def test_load_csv_success(
        self,
        mock_logger,
        mock_read_csv,
        mock_storage_client,
        mock_setup_gcp_credentials,
    ):
        # Setup
        bucket_name = "test-bucket"
        file_name = "test.csv"
        dummy_bytes = b"col1,col2\n1,2\n3,4\n"
        expected_df = pl.DataFrame({"col1": [1, 3], "col2": [2, 4]})

        # Set up the fake GCS objects.
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        # Simulate pl.read_csv returning expected_df.
        mock_read_csv.return_value = expected_df

        # Test
        result = load_bucket_data(bucket_name, file_name)

        # Assert
        assert_frame_equal(result, expected_df)
        mock_read_csv.assert_called_once()
        args, kwargs = mock_read_csv.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        self.assertEqual(args[0].getvalue(), dummy_bytes)
        expected_log = f"'{file_name}' from bucket '{bucket_name}' successfully read as CSV into DataFrame."
        mock_logger.info.assert_any_call(expected_log)

    
    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.pd.read_json')
    def test_load_bucket_json_success(self, mock_read_json, mock_storage_client, mock_setup_credentials):
        # Setup
        json_content = b'{"key": "value"}'
        fake_df = pd.DataFrame({"key": ["value"]})
        fake_df.is_empty = lambda: False

        mock_read_json.return_value = fake_df

        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = json_content

        fake_bucket = MagicMock()
        fake_bucket.blob.return_value = mock_blob


        fake_client = MagicMock()
        fake_client.get_bucket.return_value = fake_bucket
        mock_storage_client.return_value = fake_client

        # Test
        result_df = load_bucket_data("test_bucket", "data.json")

        # Assert
        mock_read_json.assert_called_once()
        mock_setup_credentials.assert_called_once()
        mock_storage_client.assert_called_once()
        pd.testing.assert_frame_equal(result_df, fake_df)



    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.pd.read_json')
    def test_load_json_exception(self, mock_read_json, mock_storage_client, mock_setup_credentials):
        # Prepare fake JSON content.
        json_content = b'{"key": "value"}'
        # Configure read_json to raise an exception.
        mock_read_json.side_effect = Exception("Error reading JSON")

        # Create a fake blob and bucket to simulate GCS behavior.
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = json_content

        fake_bucket = MagicMock()
        fake_bucket.blob.return_value = mock_blob

        # Setup the fake storage client.
        fake_client = MagicMock()
        fake_client.get_bucket.return_value = fake_bucket
        mock_storage_client.return_value = fake_client

        # Verify that the exception propagates.
        with self.assertRaises(Exception) as context:
            load_bucket_data("test_bucket", "data.json")
        self.assertIn("Error reading JSON", str(context.exception))


    # @patch("scripts.utils.setup_gcp_credentials")
    # @patch("scripts.utils.storage.Client")
    # @patch("scripts.utils.pl.read_json")
    # @patch("scripts.utils.logger")
    # def test_load_json_success(
    #     self,
    #     mock_logger,
    #     mock_read_json,
    #     mock_storage_client,
    #     mock_setup_gcp_credentials,
    # ):
    #     # Setup
    #     bucket_name = "test-bucket"
    #     file_name = "test.json"
    #     dummy_bytes = b'{"col1": [1,3], "col2": [2,4]}'

    #     expected_pl_df = pl.DataFrame({"col1": [1, 3], "col2": [2, 4]})

    #     # Set up fake GCS objects.
    #     mock_blob = MagicMock()
    #     mock_blob.download_as_string.return_value = dummy_bytes
    #     mock_bucket = MagicMock()
    #     mock_bucket.blob.return_value = mock_blob
    #     mock_client_instance = MagicMock()
    #     mock_client_instance.get_bucket.return_value = mock_bucket
    #     mock_storage_client.return_value = mock_client_instance
    #     mock_read_json.return_value = expected_pl_df

    #     # Test
    #     result = load_bucket_data(bucket_name, file_name)

    #     # Assert
    #     assert_frame_equal(result, expected_pl_df)
    #     mock_read_json.assert_called_once()
    #     expected_log = f"'{file_name}' from bucket '{bucket_name}' successfully read as JSON into DataFrame."
    #     mock_logger.info.assert_any_call(expected_log)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_unsupported_file_type(
        self, mock_logger, mock_storage_client, mock_setup_gcp_credentials
    ):
        # Setup
        bucket_name = "test-bucket"
        file_name = "test.txt"
        dummy_bytes = b"dummy content"

        # Set up fake GCS objects.
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        # Test
        with self.assertRaises(ValueError) as context:
            load_bucket_data(bucket_name, file_name)

        # Assert
        self.assertIn("Unsupported file type", str(context.exception))
        mock_logger.error.assert_any_call("Unsupported file type: txt")

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.pl.read_csv")
    @patch("scripts.utils.logger")
    def test_empty_dataframe_raises_value_error(
        self,
        mock_logger,
        mock_read_csv,
        mock_storage_client,
        mock_setup_gcp_credentials,
    ):
        # Setup
        bucket_name = "test-bucket"
        file_name = "test.csv"
        dummy_bytes = b"col1,col2\n"  # CSV header only, no data

        empty_df = pl.DataFrame()  # Simulate empty DataFrame.
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        mock_read_csv.return_value = empty_df

        # Test
        with self.assertRaises(ValueError) as context:
            load_bucket_data(bucket_name, file_name)

        # Assert
        self.assertIn("is empty", str(context.exception))
        expected_error = f"DataFrame loaded from bucket '{bucket_name}', file '{file_name}' is empty."
        mock_logger.error.assert_any_call(expected_error)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.pl.read_csv")
    @patch("scripts.utils.logger")
    def test_empty_dataframe_raises_exception(
        self,
        mock_logger,
        mock_read_csv,
        mock_storage_client,
        mock_setup_gcp_credentials,
    ):
        # Setup
        bucket_name = "dummy-bucket"
        file_name = "dummy.csv"
        dummy_bytes = b"col1,col2\n"  # header only, no data
        empty_df = pl.DataFrame()  # Empty Polars DataFrame

        # Set up fake GCS objects.
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        mock_read_csv.return_value = empty_df

        # Test
        with self.assertRaises(ValueError) as context:
            load_bucket_data(bucket_name, file_name)

        # Assert
        self.assertIn("is empty", str(context.exception))
        expected_error = f"DataFrame loaded from bucket '{bucket_name}', file '{file_name}' is empty."
        mock_logger.error.assert_any_call(expected_error)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.pl.read_csv")
    @patch("scripts.utils.logger")
    def test_csv_read_error(
        self,
        mock_logger,
        mock_read_csv,
        mock_storage_client,
        mock_setup_gcp_credentials,
    ):
        # Setup
        bucket_name = "dummy-bucket"
        file_name = "dummy.csv"
        dummy_bytes = b"some content"

        # Set up fake GCS objects.
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        # Simulate an error when reading CSV.
        mock_read_csv.side_effect = Exception("CSV read error")

        # Test
        with self.assertRaises(Exception) as context:
            load_bucket_data(bucket_name, file_name)

        # Assert
        self.assertIn("CSV read error", str(context.exception))
        expected_log = f"Error reading '{file_name}' as CSV: CSV read error"
        mock_logger.error.assert_any_call(expected_log)

    # @patch("scripts.utils.setup_gcp_credentials")
    # @patch("scripts.utils.storage.Client")
    # @patch("scripts.utils.pl.read_json")
    # @patch("scripts.utils.logger")
    # def test_json_read_error(
    #     self,
    #     mock_logger,
    #     mock_read_json,
    #     mock_storage_client,
    #     mock_setup_gcp_credentials,
    # ):
    #     # Setup
    #     bucket_name = "dummy-bucket"
    #     file_name = "dummy.json"
    #     dummy_bytes = b'{"col1": [1,2]}'

    #     # Set up fake GCS objects.
    #     mock_blob = MagicMock()
    #     mock_blob.download_as_string.return_value = dummy_bytes
    #     mock_bucket = MagicMock()
    #     mock_bucket.blob.return_value = mock_blob
    #     mock_client_instance = MagicMock()
    #     mock_client_instance.get_bucket.return_value = mock_bucket
    #     mock_storage_client.return_value = mock_client_instance

    #     # Simulate an error when reading JSON.
    #     mock_read_json.side_effect = Exception("JSON read error")

    #     # Test
    #     with self.assertRaises(Exception) as context:
    #         load_bucket_data(bucket_name, file_name)

    #     # Assert
    #     self.assertIn("JSON read error", str(context.exception))
    #     expected_log = f"Error reading '{file_name}' as JSON: JSON read error"
    #     mock_logger.error.assert_any_call(expected_log)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.pl.read_excel")
    @patch("scripts.utils.logger")
    def test_excel_read_error(
        self,
        mock_logger,
        mock_read_excel,
        mock_storage_client,
        mock_setup_gcp_credentials,
    ):
        # Setup
        bucket_name = "dummy-bucket"
        file_name = "dummy.xlsx"
        dummy_bytes = b"excel content"

        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        # Simulate an error when reading Excel.
        mock_read_excel.side_effect = Exception("Excel read error")

        # Setup
        with self.assertRaises(Exception) as context:
            load_bucket_data(bucket_name, file_name)

        # Assert
        self.assertIn("Excel read error", str(context.exception))
        expected_log = (
            f"Error reading '{file_name}' as Excel: Excel read error"
        )
        mock_logger.error.assert_any_call(expected_log)

    @patch("scripts.utils.logger")
    def test_env_not_set(self, mock_logger):
        # Test
        with patch.dict(os.environ, {}, clear=True):
            setup_gcp_credentials()
            # Assert
            self.assertEqual(
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                "/app/secret/gcp-key.json",
            )
            mock_logger.info.assert_called_with(
                "Set GCP credentials path to: /app/secret/gcp-key.json"
            )

    # @patch("scripts.utils.logger")
    # def test_setup_gcp_credentials_exception(self, mock_logger):
    #     # Patch os.environ.get to raise an exception.
    #     with patch.dict(os.environ, {}, clear=True):
    #         with patch("os.environ.get", side_effect=Exception("test error")):
    #             with self.assertRaises(Exception) as context:
    #                 setup_gcp_credentials()
    #             self.assertIn("test error", str(context.exception))

    #             # Verify that logger.error was called with an appropriate message.
    #             self.assertTrue(
    #                 any(
    #                     "Error setting up GCP credentials:" in str(arg)
    #                     for call_args, _ in mock_logger.error.call_args_list
    #                     for arg in call_args
    #                 ),
    #                 "Expected error log containing 'Error setting up GCP credentials:'",
    #             )

    @patch("scripts.utils.logger")
    def test_env_already_set(self, mock_logger):
        # Test
        with patch.dict(
            os.environ,
            {"GOOGLE_APPLICATION_CREDENTIALS": "/app/secret/gcp-key.json"},
            clear=True,
        ):
            setup_gcp_credentials()
            # Assert
            self.assertEqual(
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                "/app/secret/gcp-key.json",
            )
            mock_logger.info.assert_called_with(
                "Using existing GCP credentials from: /app/secret/gcp-key.json"
            )

    # @patch('scripts.utils.logger')
    # def test_error_setting_env_setup_cred(self, mock_logger):
    #     """Test that if setting os.environ throws an error, it is logged and re-raised."""
    #     # Clear the environment
    #     with patch.dict(os.environ, {}, clear=True):
    #         # Patch os.environ to be our ExceptionDict instance.
    #         with patch("os.environ", new=ExceptionDict()):
    #             with self.assertRaises(Exception) as context:
    #                 setup_gcp_credentials()
    #             self.assertIn("Test exception on set", str(context.exception))
    #             mock_logger.error.assert_called()

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_csv_success(
        self, mock_logger, mock_storage_client, mock_setup_gcp_credentials
    ):
        bucket_name = "test-bucket"
        destination_blob_name = "data.csv"
        # Create a dummy Polars DataFrame.
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_content = "a,b\n1,3\n2,4\n"
        with patch.object(
            df, "write_csv", return_value=csv_content
        ) as mock_write_csv:
            # Set up fake GCS objects.
            mock_blob = MagicMock()
            mock_blob.upload_from_string = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_client_instance = MagicMock()
            mock_client_instance.get_bucket.return_value = mock_bucket
            mock_storage_client.return_value = mock_client_instance

            # Call the function.
            upload_to_gcs(df, bucket_name, destination_blob_name)

            # Assert CSV branch.
            mock_write_csv.assert_called_once()
            mock_blob.upload_from_string.assert_called_once_with(
                csv_content, content_type="text/csv"
            )
            mock_logger.info.assert_any_call("CSV data uploaded successfully")
            mock_logger.info.assert_any_call(
                "Upload successful to GCS. Blob name: %s",
                destination_blob_name,
            )

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_json_standard_success(
        self, mock_logger, mock_storage_client, mock_setup_gcp_credentials
    ):
        bucket_name = "test-bucket"
        destination_blob_name = "data.json"
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        json_content = '{"a": [1, 2], "b": [3, 4]}'
        with patch.object(
            df, "write_json", return_value=json_content
        ) as mock_write_json:
            mock_blob = MagicMock()
            mock_blob.upload_from_string = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_client_instance = MagicMock()
            mock_client_instance.get_bucket.return_value = mock_bucket
            mock_storage_client.return_value = mock_client_instance

            upload_to_gcs(df, bucket_name, destination_blob_name)

            mock_write_json.assert_called_once()
            mock_blob.upload_from_string.assert_called_once_with(
                json_content, content_type="application/json"
            )
            mock_logger.info.assert_any_call(
                "JSON data uploaded successfully using write_json"
            )

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_unsupported_file_extension(
        self, mock_logger, mock_storage_client, mock_setup_gcp_credentials
    ):
        # Setup
        bucket_name = "test-bucket"
        destination_blob_name = "data.txt"
        df = pl.DataFrame({"a": [1, 2]})

        with self.assertRaises(ValueError) as context:
            upload_to_gcs(df, bucket_name, destination_blob_name)

        self.assertIn(
            "Unsupported file extension: txt", str(context.exception)
        )


    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.preprocessing.logger")
    def test_upload_csv_success_upload_to_gcs(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Setup a DataFrame and simulate write_csv returning CSV content.
        df = pl.DataFrame({"col": [1, 2, 3]})
        csv_content = "col\n1\n2\n3\n"
        df.write_csv = lambda: csv_content

        destination_blob_name = "test_file.csv"
        bucket_name = "test_bucket"

        # Setup a dummy bucket and blob.
        dummy_blob = MagicMock()
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        # Call the function.
        upload_to_gcs(df, bucket_name, destination_blob_name)

        # Assert credentials were set up and CSV branch was executed.
        mock_setup_creds.assert_called_once()
        dummy_blob.upload_from_string.assert_called_once_with(csv_content, content_type="text/csv")

    @patch("scripts.preprocessing.logger")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.setup_gcp_credentials")
    def test_unsupported_extension_upload_to_gcs(self, mock_setup_creds, mock_storage_client, mock_logger):
        # Create a dummy DataFrame.
        df = pl.DataFrame({"col": [1, 2, 3]})
        destination_blob_name = "test_file.unsupported"
        bucket_name = "test_bucket"

        # Expect a ValueError for unsupported file extensions.
        with self.assertRaises(ValueError) as context:
            upload_to_gcs(df, bucket_name, destination_blob_name)
        self.assertIn("Unsupported file extension", str(context.exception))

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.preprocessing.logger")
    def test_upload_csv_single_record_upload_to_gcs(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Create a single-record DataFrame.
        df = pl.DataFrame({"col": [42]})
        csv_content = "col\n42\n"
        df.write_csv = lambda: csv_content

        destination_blob_name = "single_record.csv"
        bucket_name = "test_bucket"

        # Setup dummy bucket and blob.
        dummy_blob = MagicMock()
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        # Call the function.
        upload_to_gcs(df, bucket_name, destination_blob_name)

        # Verify credentials and CSV branch.
        mock_setup_creds.assert_called_once()
        dummy_blob.upload_from_string.assert_called_once_with(csv_content, content_type="text/csv")

        # Verify that CSV success and final success messages are logged.
        expected_final_call = call("Upload successful to GCS. Blob name: %s", destination_blob_name)
 

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_csv_success_upload_gcs_csv(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Similar to the previous CSV test.
        df = pl.DataFrame({"col": [42]})
        csv_content = "col\n42\n"
        df.write_csv = lambda: csv_content

        destination_blob_name = "single_record.csv"
        bucket_name = "test_bucket"

        dummy_blob = MagicMock()
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        upload_to_gcs(df, bucket_name, destination_blob_name)

        mock_setup_creds.assert_called_once()
        dummy_blob.upload_from_string.assert_called_once_with(csv_content, content_type="text/csv")

        expected_start_call = call("Starting upload to GCS. Bucket: %s, Blob: %s", bucket_name, destination_blob_name)
        expected_csv_success_call = call("CSV data uploaded successfully")
        expected_final_call = call("Upload successful to GCS. Blob name: %s", destination_blob_name)
        self.assertIn(expected_start_call, mock_logger.info.call_args_list)
        self.assertIn(expected_csv_success_call, mock_logger.info.call_args_list)
        self.assertIn(expected_final_call, mock_logger.info.call_args_list)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_json_success_write_json(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Create a dummy DataFrame and simulate write_json returning valid JSON.
        df = pl.DataFrame({"col": [1, 2, 3]})
        json_content = '{"col": [1, 2, 3]}'
        df.write_json = lambda: json_content

        destination_blob_name = "test_file.json"
        bucket_name = "test_bucket"

        dummy_blob = MagicMock()
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        upload_to_gcs(df, bucket_name, destination_blob_name)

        # Verify JSON branch using write_json.
        dummy_blob.upload_from_string.assert_called_once_with(json_content, content_type="application/json")
        expected_json_success_call = call("JSON data uploaded successfully using write_json")
        expected_final_call = call("Upload successful to GCS. Blob name: %s", destination_blob_name)
        self.assertIn(expected_json_success_call, mock_logger.info.call_args_list)
        self.assertIn(expected_final_call, mock_logger.info.call_args_list)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_json_alternative_dict_conversion(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Force write_json to fail so that dict conversion is attempted.
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_json = lambda: (_ for _ in ()).throw(Exception("write_json failure"))

        destination_blob_name = "test_file.json"
        bucket_name = "test_bucket"

        dummy_blob = MagicMock()
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        # Expected dict conversion: since our DataFrame is simple, to_dict conversion works as is.
        expected_data_dict = df.to_pandas().to_dict(orient="records")
        expected_json = json.dumps(expected_data_dict, indent=2)

        upload_to_gcs(df, bucket_name, destination_blob_name)

        dummy_blob.upload_from_string.assert_called_once_with(expected_json, content_type="application/json")
        expected_log_call = call("JSON data uploaded successfully using dict conversion")
        expected_final_call = call("Upload successful to GCS. Blob name: %s", destination_blob_name)
        self.assertIn(expected_log_call, mock_logger.info.call_args_list)
        self.assertIn(expected_final_call, mock_logger.info.call_args_list)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_json_alternative_pandas_conversion(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Force write_json to fail and simulate failure in the dict conversion branch.
        df = pl.DataFrame({"col": [1, 2, 3]})
        df.write_json = lambda: (_ for _ in ()).throw(Exception("write_json failure"))

        destination_blob_name = "test_file.json"
        bucket_name = "test_bucket"

        dummy_blob = MagicMock()
        # Simulate that the first upload (dict conversion) fails, then the second attempt (pandas conversion) succeeds.
        dummy_blob.upload_from_string.side_effect = [Exception("Dict conversion failure"), None]
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        upload_to_gcs(df, bucket_name, destination_blob_name)

        # Final branch uses pandas conversion.
        expected_json = df.to_pandas().to_json(orient="records")
        # Ensure that upload_from_string was called twice.
        self.assertEqual(dummy_blob.upload_from_string.call_count, 2)
        second_call_args = dummy_blob.upload_from_string.call_args_list[1][0]
        self.assertEqual(second_call_args[0], expected_json)
        self.assertEqual(dummy_blob.upload_from_string.call_args_list[1][1]["content_type"], "application/json")
        expected_log_call = call("JSON data uploaded successfully using pandas conversion")
        expected_final_call = call("Upload successful to GCS. Blob name: %s", destination_blob_name)
        self.assertIn(expected_log_call, mock_logger.info.call_args_list)
        self.assertIn(expected_final_call, mock_logger.info.call_args_list)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.storage.Client")
    @patch("scripts.utils.logger")
    def test_upload_to_gcs_bucket_error(self, mock_logger, mock_storage_client, mock_setup_creds):
        # Simulate an error when getting the bucket.
        df = pl.DataFrame({"col": [1, 2, 3]})
        destination_blob_name = "file.csv"
        bucket_name = "test_bucket"

        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.side_effect = Exception("Bucket error")
        mock_storage_client.return_value = dummy_storage_instance

        with self.assertRaises(Exception) as context:
            upload_to_gcs(df, bucket_name, destination_blob_name)
        self.assertIn("Bucket error", str(context.exception))
        # Check that an error was logged.
        self.assertTrue(
            any("Bucket error" in str(arg) for call_args in mock_logger.error.call_args_list for arg in call_args[0]),
            "Expected an error log containing 'Bucket error'."
        )


    
    # Unit tests for send_anomaly_alert function.

    # Test case where no anomalies.
    @patch("scripts.utils.send_email")
    @patch("scripts.utils.logger")
    def test_send_anomaly_alert_no_anomalies(self, mock_logger, mock_send_email):
        # Setup: use empty DataFrames for each anomaly type.
        anomalies = {
            "price_anomalies": pl.DataFrame(),
            "quantity_anomalies": pl.DataFrame(),
        }

        # Execute the function with anomalies.
        send_anomaly_alert(anomalies=anomalies, recipient="test@example.com", subject="Alert")

        # Assert: no email should be sent, and the logger should record that no alert was sent.
        mock_send_email.assert_not_called()
        mock_logger.info.assert_any_call("No anomalies detected; no alert email sent.")


    # Test case where with anomalies.
    @patch("scripts.preprocessing.send_email")
    @patch("scripts.preprocessing.logger")
    def test_send_anomaly_alert_with_anomalies(
        self, mock_logger, mock_send_email
    ):
        # Setup
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {
            "price_anomalies": pl.DataFrame(),
            "quantity_anomalies": non_empty_df,
        }

        mock_send_email.return_value = None

        # Test
        send_anomaly_alert(
            anomalies, recipient="test@example.com", subject="Alert"
        )

        # Assert
        mock_send_email.assert_called_once()
        args, kwargs = mock_send_email.call_args
        self.assertEqual(kwargs["emailid"], "test@example.com")
        self.assertEqual(kwargs["subject"], "Alert")

        # Verify the email body matches the expected message.
        expected_body = (
            "Hi,\n\n"
            "Anomalies have been detected in the dataset. "
            "Please see the attached CSV file for details.\n\n"
            "Thank you!"
        )
        self.assertEqual(kwargs["body"], expected_body)

        attachment_df = kwargs["attachment"]
        self.assertIn("anomaly_type", attachment_df.columns)
        self.assertTrue(
            "quantity_anomalies" in attachment_df["anomaly_type"].values
        )
        mock_logger.info.assert_any_call("Anomaly alert email sent.")



    # Test case where Exception handled.
    @patch(
        "scripts.preprocessing.send_email",
        side_effect=Exception("Error sending an alert email."),
    )
    @patch("scripts.preprocessing.logger")
    def test_send_anomaly_alert_throws_exception(
        self, mock_logger, mock_send_email
    ):
        # Setup
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {"price_anomalies": non_empty_df}

        # Test
        send_anomaly_alert(
            anomalies, recipient="test@example.com", subject="Alert"
        )

        # Assert
        mock_send_email.assert_called_once()
        mock_logger.error.assert_called_with(
            "Error sending an alert email: Error sending an alert email."
        )


    
    @patch("scripts.utils.send_email")
    @patch("scripts.utils.logger")
    def test_send_anomaly_alert_with_anomalies(self, mock_logger, mock_send_email):
        # Setup
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {
            "price_anomalies": pl.DataFrame(),
            "quantity_anomalies": non_empty_df,
        }

        mock_send_email.return_value = None

        # Test
        send_anomaly_alert(anomalies=anomalies, recipient="test@example.com", subject="Alert")

        # Assert: verify send_email is called with the expected arguments.
        mock_send_email.assert_called_once()
        args, kwargs = mock_send_email.call_args
        self.assertEqual(kwargs["emailid"], "test@example.com")
        self.assertEqual(kwargs["subject"], "Alert")

        # Verify the email body matches the expected message.
        expected_body = (
            "Hi,\n\n"
            "Anomalies have been detected in the dataset. "
            "Please see the attached CSV file for details.\n\n"
            "Thank you!"
        )
        self.assertEqual(kwargs["body"], expected_body)

        # Check that the attachment has the correct anomaly_type.
        attachment_df = kwargs["attachment"]
        self.assertIn("anomaly_type", attachment_df.columns)
        self.assertTrue("quantity_anomalies" in attachment_df["anomaly_type"].values)
        mock_logger.info.assert_any_call("Anomaly alert email sent.")


    # Test case where an Exception is raised during email sending.
    @patch(
        "scripts.utils.send_email",
        side_effect=Exception("Error sending an alert email.")
    )
    @patch("scripts.utils.logger")
    def test_send_anomaly_alert_throws_exception(self, mock_logger, mock_send_email):
        # Setup
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {"price_anomalies": non_empty_df}

        # Test and Assert: Expect an exception to be raised.
        with self.assertRaises(Exception) as context:
            send_anomaly_alert(anomalies=anomalies, recipient="test@example.com", subject="Alert")

        # Verify send_email was called once.
        mock_send_email.assert_called_once()

        # Verify that the logger.error was called with the expected error message.
        mock_logger.error.assert_called_with(
            "Error sending anomaly alert: Error sending an alert email."
        )


    @patch("scripts.utils.send_email")
    @patch("scripts.utils.logger")
    def test_no_input_provided(self, mock_logger, mock_send_email):
        # Neither anomalies nor df is provided.
        send_anomaly_alert()
        
        # Expect that send_email is not called.
        mock_send_email.assert_not_called()
        # Expect a log message indicating no anomalies provided.
        mock_logger.info.assert_any_call("No anomalies provided; no alert email sent.")

    @patch("scripts.utils.send_email")
    @patch("scripts.utils.logger")
    def test_empty_anomalies_dict(self, mock_logger, mock_send_email):
        # Provide anomalies dictionary with only empty DataFrames.
        anomalies = {
            "price_anomalies": pl.DataFrame(),
            "quantity_anomalies": pl.DataFrame(),
        }
        send_anomaly_alert(anomalies=anomalies, recipient="test@example.com", subject="Alert")
        
        # Expect that send_email is not called.
        mock_send_email.assert_not_called()
        # Expect a log message indicating no anomalies detected.
        mock_logger.info.assert_any_call("No anomalies detected; no alert email sent.")

    @patch("scripts.utils.send_email")
    @patch("scripts.utils.logger")
    def test_anomalies_dict_with_nonempty(self, mock_logger, mock_send_email):
        # Setup an anomalies dictionary with one non-empty DataFrame.
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {
            "price_anomalies": pl.DataFrame(),  # empty
            "quantity_anomalies": non_empty_df,  # non-empty
        }
        send_anomaly_alert(anomalies=anomalies, recipient="test@example.com", subject="Alert")
        
        # Verify that send_email is called once.
        mock_send_email.assert_called_once()
        # Retrieve the call arguments to inspect the attachment.
        args, kwargs = mock_send_email.call_args
        
        self.assertEqual(kwargs["emailid"], "test@example.com")
        self.assertEqual(kwargs["subject"], "Alert")
        # Check the default message was used.
        expected_body = (
            "Hi,\n\n"
            "Anomalies have been detected in the dataset. "
            "Please see the attached CSV file for details.\n\n"
            "Thank you!"
        )
        self.assertEqual(kwargs["body"], expected_body)
        # The attachment should be a DataFrame with an extra column 'anomaly_type'.
        attachment_df = kwargs["attachment"]
        self.assertIn("anomaly_type", attachment_df.columns)
        # Check that the non-empty anomaly's type was added.
        self.assertTrue("quantity_anomalies" in attachment_df["anomaly_type"].values)
        mock_logger.info.assert_any_call("Anomaly alert email sent.")


    @patch("scripts.utils.send_email")
    @patch("scripts.utils.logger")
    def test_direct_dataframe_mode(self, mock_logger, mock_send_email):
        # Provide a non-empty DataFrame directly via the df parameter.
        data = {"col": [10, 20, 30]}
        direct_df = pd.DataFrame(data)
        send_anomaly_alert(df=direct_df, recipient="direct@example.com", subject="Direct Alert")
        
        # Expect send_email to be called with the provided DataFrame.
        mock_send_email.assert_called_once()
        args, kwargs = mock_send_email.call_args
        
        # The recipient is passed as the first positional argument.
        self.assertEqual(args[0], "direct@example.com")
        self.assertEqual(kwargs["subject"], "Direct Alert")
        expected_body = (
            "Hi,\n\n"
            "Anomalies have been detected in the dataset. "
            "Please see the attached CSV file for details.\n\n"
            "Thank you!"
        )
        self.assertEqual(kwargs["body"], expected_body)
        # The attachment should be exactly the direct DataFrame.
        pd.testing.assert_frame_equal(kwargs["attachment"], direct_df)
        # Check logger message for direct mode.
        mock_logger.info.assert_any_call("Data Validation Anomaly alert sent to user: direct@example.com")


    @patch("scripts.utils.send_email", side_effect=Exception("Simulated email error"))
    @patch("scripts.utils.logger")
    def test_exception_in_send_email(self, mock_logger, mock_send_email):
        # Setup an anomalies dictionary with a non-empty DataFrame.
        data = {"col": [5, 6, 7]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {"price_anomalies": non_empty_df}
        
        with self.assertRaises(Exception) as context:
            send_anomaly_alert(anomalies=anomalies, recipient="error@example.com", subject="Error Alert")
        
        # Ensure send_email was called.
        mock_send_email.assert_called_once()
        # Verify that the error was logged.
        mock_logger.error.assert_called_with("Error sending anomaly alert: Simulated email error")
        # Optionally, check that the raised exception message matches.
        self.assertEqual(str(context.exception), "Simulated email error")


    
    def test_no_missing_columns(self):
        # Test when there are no missing columns.
        df = pd.DataFrame({"a": [1, 2, 3]})
        error_indices = set()
        error_reasons = {}
        missing_columns = []
        
        collect_validation_errors(df, missing_columns, error_indices, error_reasons)
        
        # Expect no changes to error_indices or error_reasons.
        self.assertEqual(len(error_indices), 0)
        self.assertEqual(len(error_reasons), 0)

    def test_empty_dataframe_with_missing_columns(self):
        # Test when missing_columns is non-empty but the DataFrame is empty.
        df = pd.DataFrame({"a": []})
        error_indices = set()
        error_reasons = {}
        missing_columns = ["col1"]
        
        collect_validation_errors(df, missing_columns, error_indices, error_reasons)
        
        # Even though missing_columns is non-empty, the DataFrame has no rows.
        self.assertEqual(len(error_indices), 0)
        self.assertEqual(len(error_reasons), 0)

    def test_dataframe_with_missing_columns(self):
        # Test when the DataFrame has rows and missing_columns is non-empty.
        df = pd.DataFrame({"a": [1, 2, 3]})
        error_indices = set()
        error_reasons = {}
        missing_columns = ["colA", "colB"]
        
        collect_validation_errors(df, missing_columns, error_indices, error_reasons)
        
        # Expect error_indices to contain indices for all rows.
        self.assertEqual(error_indices, {0, 1, 2})
        
        # Each error reason should match the expected message.
        expected_message = "Missing columns: colA, colB"
        for idx in range(len(df)):
            self.assertIn(idx, error_reasons)
            self.assertEqual(error_reasons[idx], [expected_message])




    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.logger")
    @patch("scripts.utils.storage.Client")
    def test_delete_blob_success(self, mock_storage_client, mock_logger, mock_setup):
        bucket_name = "test_bucket"
        blob_name = "test_blob"
        
        # Create a mock blob and bucket
        mock_blob = Mock()
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket
        
        # Call the function under test
        result = delete_blob_from_bucket(bucket_name, blob_name)
        
        # Assert the chain of calls:
        # 1. bucket.blob(blob_name) was called
        mock_bucket.blob.assert_called_with(blob_name)
        # 2. blob.delete() was called
        mock_blob.delete.assert_called_once()
        # 3. logger.info was called with the expected message
        mock_logger.info.assert_called_with(f"Blob {blob_name} deleted from bucket {bucket_name}")
        # 4. The function returns True
        self.assertTrue(result)

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.logger")
    @patch("scripts.utils.storage.Client")
    def test_delete_blob_failure(self, mock_storage_client, mock_logger, mock_setup):
        bucket_name = "test_bucket"
        blob_name = "test_blob"
        
        # Create a mock blob that raises an Exception on delete
        mock_blob = Mock()
        mock_blob.delete.side_effect = Exception("Delete error")
        mock_bucket = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket
        
        # Call the function under test; expect it to handle the exception and return False.
        result = delete_blob_from_bucket(bucket_name, blob_name)
        
        # Assert that logger.error was called with a message containing the exception text.
        mock_logger.error.assert_called_with(
            f"Error deleting blob {blob_name} from bucket {bucket_name}: Delete error"
        )
        # The function should return False on exception.
        self.assertFalse(result)


    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.logger")
    @patch("scripts.utils.storage.Client")
    def test_list_bucket_blobs_success(self, mock_storage_client, mock_logger, mock_setup):
        bucket_name = "test_bucket"
        
        # Create mock blob objects with a 'name' attribute.
        mock_blob1 = Mock()
        mock_blob1.name = "blob1"
        mock_blob2 = Mock()
        mock_blob2.name = "blob2"
        blobs = [mock_blob1, mock_blob2]
        
        # Configure the mock bucket to return the blobs.
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = blobs
        
        # Configure the storage client to return the mock bucket.
        mock_storage_client.return_value.get_bucket.return_value = mock_bucket
        
        # Call the function under test.
        result = list_bucket_blobs(bucket_name)
        
        # Assert that the result is a list of blob names.
        self.assertEqual(result, ["blob1", "blob2"])
        
        # Verify that the logger.info was called with the expected message.
        mock_logger.info.assert_called_with("Found 2 files in bucket 'test_bucket'")

    @patch("scripts.utils.setup_gcp_credentials")
    @patch("scripts.utils.logger")
    @patch("scripts.utils.storage.Client")
    def test_list_bucket_blobs_exception(self, mock_storage_client, mock_logger, mock_setup):
        bucket_name = "test_bucket"
        
        # Configure the storage client to raise an Exception when getting the bucket.
        mock_storage_client.return_value.get_bucket.side_effect = Exception("Test error")
        
        with self.assertRaises(Exception) as context:
            list_bucket_blobs(bucket_name)
        
        # Verify that the logger.error was called with an error message containing the exception text.
        mock_logger.error.assert_called_with("Error listing blobs in bucket 'test_bucket': Test error")
        # Optionally, you can also assert that the raised exception message is correct.
        self.assertEqual(str(context.exception), "Test error")



if __name__ == "__main__":
    unittest.main()
