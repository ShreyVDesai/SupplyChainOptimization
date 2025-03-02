import io
import unittest
from unittest.mock import patch, MagicMock
from scripts.logger import logger
import polars as pl
import pandas as pd
from polars.testing import assert_frame_equal
from scripts.utils import load_bucket_data, send_email, upload_to_gcs, setup_gcp_credentials
from google.cloud import storage
import logging
import os


class TestUtils(unittest.TestCase):

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

class TestLoadBucketData(unittest.TestCase):
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_csv_success(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.csv"
        
        # Create sample CSV data
        csv_content = b"Date,Unit Price,Quantity,Transaction ID,Store Location,Product Name,Producer ID\n2023-01-01,10.99,5,12345,New York,Product A,P001"
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = csv_content
        
        # Act
        result = load_bucket_data(bucket_name, file_name)
        
        # Assert
        mock_setup_credentials.assert_called_once()
        mock_client.return_value.get_bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(file_name)
        mock_blob.download_as_string.assert_called_once()
        
        # Check that result is a polars DataFrame with the expected structure
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(len(result), 1)  # One row
        self.assertEqual(result.shape[1], 7)  # Seven columns
        self.assertEqual(list(result.columns), [
            "Date", "Unit Price", "Quantity", "Transaction ID", 
            "Store Location", "Product Name", "Producer ID"
        ])
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_xlsx_success(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.xlsx"
        
        # Mock Excel file content
        excel_content = b"dummy excel binary content"
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = excel_content
        
        # Mock the polars read_excel function
        mock_df = pl.DataFrame({
            "Date": ["2023-01-01"],
            "Unit Price": [10.99],
            "Quantity": [5],
            "Transaction ID": ["12345"],
            "Store Location": ["New York"],
            "Product Name": ["Product A"],
            "Producer ID": ["P001"]
        })
        
        with patch('scripts.utils.pl.read_excel', return_value=mock_df):
            # Act
            result = load_bucket_data(bucket_name, file_name)
            
            # Assert
            mock_setup_credentials.assert_called_once()
            self.assertIsInstance(result, pl.DataFrame)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.shape[1], 7)
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_json_success(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.json"
        
        # Mock JSON content
        json_content = b'[{"Date":"2023-01-01","Unit Price":10.99,"Quantity":5,"Transaction ID":"12345","Store Location":"New York","Product Name":"Product A","Producer ID":"P001"}]'
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = json_content
        
        # Mock pandas DataFrame
        pd_df = pl.DataFrame({
            "Date": ["2023-01-01"],
            "Unit Price": [10.99],
            "Quantity": [5],
            "Transaction ID": ["12345"],
            "Store Location": ["New York"],
            "Product Name": ["Product A"],
            "Producer ID": ["P001"]
        })
        
        with patch('scripts.utils.pl.read_json', return_value=pd_df):
            # Act
            result = load_bucket_data(bucket_name, file_name)
            
            # Assert
            mock_setup_credentials.assert_called_once()
            self.assertIsInstance(result, pl.DataFrame)
            # Additional assertions as needed
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_unsupported_extension(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.txt"
        
        # Mock blob content
        blob_content = b"some text content"
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = blob_content
        
        # Act and Assert
        with self.assertRaises(ValueError) as context:
            load_bucket_data(bucket_name, file_name)
        
        self.assertIn("Unsupported file type: txt", str(context.exception))
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_empty_dataframe(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.csv"
        
        # Mock CSV with headers only
        csv_content = b"Date,Unit Price,Quantity,Transaction ID,Store Location,Product Name,Producer ID"
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = csv_content
        
        # Mock an empty DataFrame
        with patch('scripts.utils.pl.read_csv', return_value=pl.DataFrame()):
            # Act and Assert
            with self.assertRaises(ValueError) as context:
                load_bucket_data(bucket_name, file_name)
            
            self.assertIn("is empty", str(context.exception))
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_missing_columns(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.csv"
        
        # Mock CSV with missing columns
        csv_content = b"Date,Unit Price,Quantity,Transaction ID"
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = csv_content
        
        # Mock a DataFrame with missing columns
        mock_df = pl.DataFrame({
            "Date": ["2023-01-01"],
            "Unit Price": [10.99],
            "Quantity": [5],
            "Transaction ID": ["12345"]
        })
        
        with patch('scripts.utils.pl.read_csv', return_value=mock_df):
            # Act and Assert
            with self.assertRaises(ValueError) as context:
                load_bucket_data(bucket_name, file_name)
            
            self.assertIn("missing required columns", str(context.exception))
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_bucket_error(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.csv"
        
        # Mock a storage error
        mock_client.return_value.get_bucket.side_effect = Exception("Bucket not found")
        
        # Act and Assert
        with self.assertRaises(Exception) as context:
            load_bucket_data(bucket_name, file_name)

        self.assertIn("Bucket not found", str(context.exception))
    
    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    def test_load_bucket_data_csv_read_error(self, mock_setup_credentials, mock_client):
        # Arrange
        bucket_name = "test-bucket"
        file_name = "test-file.csv"
        
        # Mock blob content
        csv_content = b"invalid,csv,content"
        
        # Mock the storage client, bucket, and blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = csv_content
        
        # Mock CSV reading error
        with patch('scripts.utils.pl.read_csv', side_effect=Exception("CSV parsing error")):
            # Act and Assert
            with self.assertRaises(Exception) as context:
                load_bucket_data(bucket_name, file_name)
            
            self.assertIn("CSV parsing error", str(context.exception))

class TestUploadToGCS(unittest.TestCase):
    """Test cases for the upload_to_gcs function."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample DataFrame for testing
        self.test_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        self.bucket_name = "test-bucket"

    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.logger.info')
    def test_upload_csv_format(self, mock_logger_info, mock_setup_credentials, mock_client):
        """Test uploading a DataFrame in CSV format."""
        # Set up mocks
        mock_client_instance = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Configure mock chain
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock DataFrame method
        self.test_df.write_csv = MagicMock(return_value="id,name\n1,Alice\n2,Bob\n3,Charlie")
        
        # Call the function with CSV file extension
        upload_to_gcs(self.test_df, self.bucket_name, "data.csv")
        
        # Verify function behavior
        mock_setup_credentials.assert_called_once()
        mock_client_instance.get_bucket.assert_called_once_with(self.bucket_name)
        mock_bucket.blob.assert_called_once_with("data.csv")
        self.test_df.write_csv.assert_called_once()
        mock_blob.upload_from_string.assert_called_once_with(
            "id,name\n1,Alice\n2,Bob\n3,Charlie", 
            content_type="text/csv"
        )
        # Verify logger was called
        assert mock_logger_info.call_count == 2

    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.logger.info')
    def test_upload_json_format(self, mock_logger_info, mock_setup_credentials, mock_client):
        """Test uploading a DataFrame in JSON format."""
        # Set up mocks
        mock_client_instance = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Configure mock chain
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock DataFrame method
        json_data = '[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"},{"id":3,"name":"Charlie"}]'
        self.test_df.write_json = MagicMock(return_value=json_data)
        
        # Call the function with JSON file extension
        upload_to_gcs(self.test_df, self.bucket_name, "data.json")
        
        # Verify function behavior
        mock_setup_credentials.assert_called_once()
        mock_client_instance.get_bucket.assert_called_once_with(self.bucket_name)
        mock_bucket.blob.assert_called_once_with("data.json")
        self.test_df.write_json.assert_called_once()
        mock_blob.upload_from_string.assert_called_once_with(
            json_data, 
            content_type="application/json"
        )
        # Verify logger was called
        assert mock_logger_info.call_count == 2

    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.logger.error')
    def test_unsupported_file_format(self, mock_logger_error, mock_setup_credentials, mock_client):
        """Test uploading with unsupported file format raises ValueError."""
        # Set up mocks
        mock_client_instance = MagicMock()
        mock_bucket = MagicMock()
        
        # Configure mock chain
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_bucket.return_value = mock_bucket
        
        # Call the function with unsupported file extension
        with self.assertRaises(ValueError) as context:
            upload_to_gcs(self.test_df, self.bucket_name, "data.xlsx")
        
        self.assertEqual(str(context.exception), "Unsupported file extension: xlsx")
        mock_setup_credentials.assert_called_once()

    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.logger.error')
    def test_gcs_connection_error(self, mock_logger_error, mock_setup_credentials, mock_client):
        """Test handling of GCS connection errors."""
        # Configure mock to raise an exception
        mock_client.side_effect = Exception("Connection failed")
        
        # Call the function and expect exception to be re-raised
        with self.assertRaises(Exception) as context:
            upload_to_gcs(self.test_df, self.bucket_name, "data.csv")
        
        self.assertEqual(str(context.exception), "Connection failed")
        mock_setup_credentials.assert_called_once()
        mock_logger_error.assert_called_once()

    @patch('scripts.utils.storage.Client')
    @patch('scripts.utils.setup_gcp_credentials')
    @patch('scripts.utils.logger.error')
    def test_bucket_not_found(self, mock_logger_error, mock_setup_credentials, mock_client):
        """Test handling of bucket not found errors."""
        # Set up mocks
        mock_client_instance = MagicMock()
        
        # Configure mock chain
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_bucket.side_effect = Exception("Bucket not found")
        
        # Call the function and expect exception to be re-raised
        with self.assertRaises(Exception) as context:
            upload_to_gcs(self.test_df, self.bucket_name, "data.csv")
        
        self.assertEqual(str(context.exception), "Bucket not found")
        mock_setup_credentials.assert_called_once()
        mock_logger_error.assert_called_once()


class TestSetupGCPCredentials(unittest.TestCase):
    def setUp(self):
        # Backup current environment
        self.original_env = os.environ.copy()

    def tearDown(self):
        # Restore environment variables to avoid side-effects across tests
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('os.path.abspath')
    @patch('os.path.exists')
    def test_primary_path_exists(self, mock_exists, mock_abspath):
        """
        Test when the primary path (project_root/secret/gcp-key.json) exists.
        """
        # Simulate __file__ path so that:
        #   script_dir = /project/my_module/subdir
        #   project_root = /project/my_module
        mock_abspath.return_value = "/project/Data-Pipeline/scripts/utils.py"

        # Define side effect for os.path.exists: primary path exists.
        def exists_side_effect(path):
            if path == "/project/secret/gcp-key.json":
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        setup_gcp_credentials()
        expected = "secret/gcp-key.json"
        self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], expected)

    @patch('os.path.abspath')
    @patch('os.path.exists')
    def test_alternative_path_exists(self, mock_exists, mock_abspath):
        """
        Test when the primary key file is missing but the alternative location exists.
        """
        mock_abspath.return_value = "/project/Data-Pipeline/scripts/utils.py"

        def exists_side_effect(path):
            # Primary location does not exist
            if path == "/project/secret/gcp-key.json":
                return False
            # The alternative path should be:
            # os.path.join(os.path.dirname(script_dir), "..", "secret", "gcp-key.json")
            if os.path.normpath(path) == os.path.normpath("/project/secret/gcp-key.json"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        setup_gcp_credentials()
        expected = os.path.normpath("/project/secret/gcp-key.json")
        self.assertEqual(os.path.normpath(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]), os.path.normpath(expected))


    @patch('os.path.abspath')
    @patch('os.path.exists')
    def test_fallback_path_exists(self, mock_exists, mock_abspath):
        """
        Test when neither the primary nor the alternative file exists but the fallback exists.
        """
        mock_abspath.return_value = "/project/Data-Pipeline/scripts/utils.py"

        def exists_side_effect(path):
            # Both primary and alt do not exist
            if path in ["/project/secret/gcp-key.json", os.path.normpath("/project/secret/gcp-key.json")]:
                return False
            # Fallback file "secret/gcp-key.json" exists
            if path == "secret/gcp-key.json":
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        setup_gcp_credentials()
        self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "secret/gcp-key.json")

if __name__ == '__main__':
    unittest.main()