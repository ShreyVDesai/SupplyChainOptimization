import io
import unittest
from unittest.mock import patch, MagicMock
from scripts.logger import logger
import polars as pl
import pandas as pd
from polars.testing import assert_frame_equal
from scripts.utils import load_bucket_data, send_email

class TestUtils(unittest.TestCase):

    @patch('scripts.utils.logger')
    @patch('scripts.utils.pl.read_excel')
    @patch('scripts.utils.storage.Client')
    def test_load_bucket_data_success(self, mock_storage_client, mock_read_excel, mock_logger):
        # Setup
        dummy_bytes = b'dummy excel bytes'
        expected_df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        })
        
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = dummy_bytes
        
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        mock_read_excel.return_value = expected_df
        bucket_name = 'test-bucket'
        file_name = 'test-file.xlsx'

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

        expected_log = f"'{file_name}' from bucket '{bucket_name}' successfully read into DataFrame."
        mock_logger.info.assert_called_once_with(expected_log)

        assert_frame_equal(result, expected_df)


    @patch('scripts.utils.logger')
    @patch('scripts.utils.storage.Client')
    def test_load_bucket_data_failure_exception_handling(self, mock_storage_client, mock_logger):
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
        
        # Set up a mock client instance whose get_bucket returns the mock bucket.
        mock_client_instance = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance
        
        bucket_name = "test-bucket"
        file_name = "test-file.xlsx"
        
        with self.assertRaises(Exception) as context:
            load_bucket_data(bucket_name, file_name)
        
        # Verify the exception message contains our simulated error.
        self.assertIn(error_message, str(context.exception))
        
        # Verify that logger.error was called with a message that includes the error.
        mock_logger.error.assert_called_once()
        error_log_msg = mock_logger.error.call_args[0][0]
        self.assertIn(error_message, error_log_msg)


    @patch('scripts.utils.logger')
    @patch('scripts.utils.pl.read_excel')
    @patch('scripts.utils.storage.Client')
    def test_empty_dataframe_raises_value_error(self, mock_storage_client, mock_read_excel, mock_logger):
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
        error_messages = [args[0][0] for args in mock_logger.error.call_args_list]
        self.assertTrue(any(expected_substring in message for message in error_messages),
                        f"None of the error messages contained '{expected_substring}'.")
        

    @patch('scripts.utils.logger')
    @patch('scripts.utils.pl.read_excel')
    @patch('scripts.utils.storage.Client')
    def test_missing_columns_raises_value_error(self, mock_storage_client, mock_read_excel, mock_logger):
        """
        Test that if the DataFrame is missing required columns, a ValueError is raised.
        """
        # Setup
        df_missing_cols = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Transaction ID": ["T123", "T456"]
        })
        mock_read_excel.return_value = df_missing_cols

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
        self.assertIn("is missing required columns", str(context.exception))

        missing_cols = ["Quantity", "Store Location", "Product Name", "Producer ID"]

        # Assert
        expected_substring = f"is missing required columns: {missing_cols}"
        error_messages = [args[0][0] for args in mock_logger.error.call_args_list]
        self.assertTrue(any(expected_substring in message for message in error_messages),
                        f"None of the error messages contained '{expected_substring}'.")
        


    ### Unit tests for send_email


    # Test case where send_email executes successfully without attachment.
    @patch('scripts.utils.logger')
    @patch('scripts.utils.smtplib.SMTP')
    def test_send_email_success_without_attachment(self, mock_smtp, mock_logger):
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
        smtp_instance.login.assert_called_once_with("talksick530@gmail.com", "celm dfaq qllh ymjv")
        smtp_instance.send_message.assert_called_once()
        mock_logger.info.assert_any_call(f"Email sent successfully to: {emailid}")


    
    # Test case where send_email executes successfully with attachment.
    @patch('scripts.utils.logger')
    @patch('scripts.utils.smtplib.SMTP')
    def test_send_email_success_with_attachment(self, mock_smtp, mock_logger):
        # Setup
        smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = smtp_instance

        emailid = "recipient@example.com"
        body = "This is a test email with attachment."
        subject = "Test Email with Attachment"

        attachment_df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        })

        # Test
        send_email(emailid, body, subject=subject, attachment=attachment_df)

        # Assert
        smtp_instance.send_message.assert_called_once()
        sent_msg = smtp_instance.send_message.call_args[0][0]
        self.assertTrue(sent_msg.is_multipart(), "EmailMessage should be multipart when an attachment is added.")    
        attachments = [part for part in sent_msg.iter_attachments()]
        self.assertEqual(len(attachments), 1, "There should be exactly one attachment.")
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
    @patch('scripts.utils.logger')
    @patch('scripts.utils.smtplib.SMTP')
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