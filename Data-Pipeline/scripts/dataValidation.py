import unittest
from unittest.mock import MagicMock, patch
import io
import logging
import polars as pl
from dataPreprocessing import load_bucket_data, detect_anomalies, upload_df_to_gcs
from datetime import datetime
from google.cloud.exceptions import GoogleCloudError

class TestLoadBucketData(unittest.TestCase):
    def setUp(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create mock data for tests
        self.mock_bucket_name = "test-bucket"
        self.mock_file_name = "test-file.xlsx"
        
        # Create a sample DataFrame for testing
        self.sample_df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    @patch('dataPreprocessing.storage.Client')
    @patch('dataPreprocessing.pl.read_excel')
    def test_load_bucket_data_success(self, mock_read_excel, mock_client):
        # Mock the storage client and related objects
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Configure the mock objects
        mock_client.return_value = mock_storage_client
        mock_storage_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = b"mock binary content"
        
        # Setup the mock to return our sample DataFrame
        mock_read_excel.return_value = self.sample_df
        
        # Call the function
        result = load_bucket_data(self.mock_bucket_name, self.mock_file_name)
        
        # Assert function called the expected methods
        mock_storage_client.get_bucket.assert_called_once_with(self.mock_bucket_name)
        mock_bucket.blob.assert_called_once_with(self.mock_file_name)
        mock_blob.download_as_string.assert_called_once()
        
        # Assert read_excel was called with BytesIO object containing the blob content
        mock_read_excel.assert_called_once()
        # Check the first argument of the first call is a BytesIO object
        args, _ = mock_read_excel.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        
        # Assert the return value is correct - proper way to compare Polars DataFrames
        self.assertTrue(result.equals(self.sample_df))

    @patch('dataPreprocessing.storage.Client')
    def test_load_bucket_data_bucket_error(self, mock_client):
        # Mock to raise an exception when getting bucket
        mock_storage_client = MagicMock()
        mock_client.return_value = mock_storage_client
        mock_storage_client.get_bucket.side_effect = Exception("Bucket not found")
        
        # Call the function and assert it raises an exception
        with self.assertRaises(Exception):
            load_bucket_data(self.mock_bucket_name, self.mock_file_name)
        
        # Verify method was called
        mock_storage_client.get_bucket.assert_called_once_with(self.mock_bucket_name)

    @patch('dataPreprocessing.storage.Client')
    def test_load_bucket_data_blob_error(self, mock_client):
        # Mock storage client
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.return_value = mock_storage_client
        mock_storage_client.get_bucket.return_value = mock_bucket
        
        # Mock to raise an exception when getting blob
        mock_bucket.blob.side_effect = Exception("Blob error")
        
        # Call the function and assert it raises an exception
        with self.assertRaises(Exception):
            load_bucket_data(self.mock_bucket_name, self.mock_file_name)
        
        # Verify methods were called
        mock_storage_client.get_bucket.assert_called_once_with(self.mock_bucket_name)
        mock_bucket.blob.assert_called_once_with(self.mock_file_name)

    @patch('dataPreprocessing.storage.Client')
    def test_load_bucket_data_download_error(self, mock_client):
        # Mock storage client and bucket
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value = mock_storage_client
        mock_storage_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock to raise an exception when downloading
        mock_blob.download_as_string.side_effect = Exception("Download error")
        
        # Call the function and assert it raises an exception
        with self.assertRaises(Exception):
            load_bucket_data(self.mock_bucket_name, self.mock_file_name)
        
        # Verify methods were called
        mock_storage_client.get_bucket.assert_called_once_with(self.mock_bucket_name)
        mock_bucket.blob.assert_called_once_with(self.mock_file_name)
        mock_blob.download_as_string.assert_called_once()

    @patch('dataPreprocessing.storage.Client')
    @patch('dataPreprocessing.pl.read_excel')
    def test_load_bucket_data_read_error(self, mock_read_excel, mock_client):
        # Mock storage client and related objects
        mock_storage_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value = mock_storage_client
        mock_storage_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.download_as_string.return_value = b"mock binary content"
        
        # Mock to raise an exception when reading the Excel file
        mock_read_excel.side_effect = Exception("Error reading Excel file")
        
        # Call the function and assert it raises an exception
        with self.assertRaises(Exception):
            load_bucket_data(self.mock_bucket_name, self.mock_file_name)
        
        # Verify methods were called
        mock_storage_client.get_bucket.assert_called_once_with(self.mock_bucket_name)
        mock_bucket.blob.assert_called_once_with(self.mock_file_name)
        mock_blob.download_as_string.assert_called_once()
        mock_read_excel.assert_called_once()

class TestDetectAnomalies(unittest.TestCase):
    
    def setUp(self):
        """Set up test data that will be used across multiple tests"""
        # Create sample transaction data
        self.test_data = pl.DataFrame({
            'Transaction ID': range(1, 21),
            'Date': [
                # Normal business hours transactions
                datetime(2023, 1, 1, 10, 0), datetime(2023, 1, 1, 14, 0), 
                datetime(2023, 1, 1, 16, 0), datetime(2023, 1, 1, 18, 0),
                datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 15, 0),
                # Late night transactions (time anomalies)
                datetime(2023, 1, 1, 2, 0), datetime(2023, 1, 1, 23, 30),
                # Next day transactions
                datetime(2023, 1, 2, 10, 0), datetime(2023, 1, 2, 14, 0),
                datetime(2023, 1, 2, 16, 0), datetime(2023, 1, 2, 18, 0),
                datetime(2023, 1, 2, 12, 0), datetime(2023, 1, 2, 15, 0),
                # Another late night transaction
                datetime(2023, 1, 2, 3, 0),
                # More normal hours
                datetime(2023, 1, 3, 10, 0), datetime(2023, 1, 3, 12, 0),
                datetime(2023, 1, 3, 14, 0), datetime(2023, 1, 3, 16, 0),
                # Format anomaly timing
                datetime(2023, 1, 3, 11, 0)
            ],
            'Product Name': [
                'Apple', 'Apple', 'Apple', 'Apple',
                'Banana', 'Banana', 'Banana', 'Banana',
                'Apple', 'Apple', 'Apple', 'Apple',
                'Banana', 'Banana', 'Banana',
                'Cherry', 'Cherry', 'Cherry', 'Cherry',
                'Cherry'
            ],
            'Unit Price': [
                # Day 1 Apples - One price anomaly (100)
                10.0, 9.5, 100.0, 10.5,
                # Day 1 Bananas - Normal prices
                2.0, 2.1, 1.9, 2.2,
                # Day 2 Apples - Normal prices
                10.2, 9.8, 10.3, 9.9,
                # Day 2 Bananas - One price anomaly (0.5)
                2.0, 0.5, 2.1,
                # Day 3 Cherries - One format anomaly (0)
                5.0, 5.2, 5.1, 4.9,
                0.0  # Invalid price for format anomaly testing
            ],
            'Quantity': [
                # Day 1 Apples - Normal quantities
                2, 3, 1, 2,
                # Day 1 Bananas - One quantity anomaly (20)
                5, 4, 20, 3,
                # Day 2 Apples - Normal quantities
                2, 1, 3, 2,
                # Day 2 Bananas - Normal quantities
                4, 3, 5,
                # Day 3 Cherries - One normal, one quantity anomaly (30), two normal
                2, 30, 3, 1,
                0  # Invalid quantity for format anomaly testing
            ]
        })
        
        # Create an empty dataframe with the same schema for edge case testing
        self.empty_data = pl.DataFrame(schema={
            'Transaction ID': pl.Int64,
            'Date': pl.Datetime,
            'Product Name': pl.Utf8,
            'Unit Price': pl.Float64,
            'Quantity': pl.Int64
        })
        
        # Create a small dataframe with insufficient data points for IQR analysis
        self.small_data = pl.DataFrame({
            'Transaction ID': [1, 2, 3],
            'Date': [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 1, 1, 14, 0),
                datetime(2023, 1, 1, 16, 0)
            ],
            'Product Name': ['Apple', 'Apple', 'Apple'],
            'Unit Price': [10.0, 9.5, 10.5],
            'Quantity': [2, 3, 2]
        })

    def test_price_anomalies(self):
        """Test detection of price anomalies"""
        # Run the function
        anomalies, clean_df = detect_anomalies(self.test_data)
        
        # Check that price anomalies were detected
        self.assertIn('price_anomalies', anomalies)
        price_anomalies = anomalies['price_anomalies']

        print(price_anomalies)
        
        # Should detect 2 price anomalies: Apple with price 100.0 and Banana with price 0.5
        self.assertEqual(len(price_anomalies), 2)
        
        # Verify specific anomalies
        transaction_ids = price_anomalies['Transaction ID'].to_list()
        self.assertIn(3, transaction_ids)  # Apple with price 100.0
        
        # Verify these anomalies are not in the clean data
        clean_transaction_ids = clean_df['Transaction ID'].to_list()
        self.assertNotIn(3, clean_transaction_ids)

    def test_quantity_anomalies(self):
        """Test detection of quantity anomalies"""
        # Run the function
        anomalies, clean_df = detect_anomalies(self.test_data)
        
        # Check that quantity anomalies were detected
        self.assertIn('quantity_anomalies', anomalies)
        quantity_anomalies = anomalies['quantity_anomalies']

        print(f"quantity: {quantity_anomalies}")
        
        # Should detect 2 quantity anomalies: Banana with quantity 20 and Cherry with quantity 30
        self.assertEqual(len(quantity_anomalies), 2)
        
        # Verify specific anomalies
        transaction_ids = quantity_anomalies['Transaction ID'].to_list()
        self.assertIn(7, transaction_ids)  # Banana with quantity 20
        self.assertIn(17, transaction_ids)  # Cherry with quantity 30
        
        # Verify these anomalies are not in the clean data
        clean_transaction_ids = clean_df['Transaction ID'].to_list()
        self.assertNotIn(7, clean_transaction_ids)
        self.assertNotIn(17, clean_transaction_ids)

    def test_time_anomalies(self):
        """Test detection of time pattern anomalies"""
        # Run the function
        anomalies, clean_df = detect_anomalies(self.test_data)
        
        # Check that time anomalies were detected
        self.assertIn('time_anomalies', anomalies)
        time_anomalies = anomalies['time_anomalies']
        
        # Should detect 3 time anomalies: two at 2:00 AM, one at 3:00 AM, and one at 11:30 PM
        self.assertEqual(len(time_anomalies), 3)
        
        # Verify specific anomalies
        transaction_ids = time_anomalies['Transaction ID'].to_list()
        self.assertIn(7, transaction_ids)  # Transaction at 2:00 AM
        self.assertIn(8, transaction_ids)  # Transaction at 11:30 PM
        self.assertIn(15, transaction_ids)  # Transaction at 3:00 AM
        
        # Verify these anomalies are not in the clean data
        clean_transaction_ids = clean_df['Transaction ID'].to_list()
        self.assertNotIn(7, clean_transaction_ids)
        self.assertNotIn(8, clean_transaction_ids)
        self.assertNotIn(15, clean_transaction_ids)

    def test_format_anomalies(self):
        """Test detection of format anomalies (invalid values)"""
        # Run the function
        anomalies, clean_df = detect_anomalies(self.test_data)
        
        # Check that format anomalies were detected
        self.assertIn('format_anomalies', anomalies)
        format_anomalies = anomalies['format_anomalies']
        
        # Should detect 1 format anomaly: Cherry with price 0.0 and quantity 0
        self.assertEqual(len(format_anomalies), 1)
        
        # Verify specific anomaly
        transaction_ids = format_anomalies['Transaction ID'].to_list()
        self.assertIn(20, transaction_ids)  # Cherry with invalid values
        
        # Verify this anomaly is not in the clean data
        clean_transaction_ids = clean_df['Transaction ID'].to_list()
        self.assertNotIn(20, clean_transaction_ids)

    def test_empty_dataframe(self):
        """Test function behavior with an empty dataframe"""
        # Run the function with empty data
        anomalies, clean_df = detect_anomalies(self.empty_data)
        
        # All anomaly categories should exist but be empty
        self.assertIn('price_anomalies', anomalies)
        self.assertEqual(len(anomalies['price_anomalies']), 0)
        
        self.assertIn('quantity_anomalies', anomalies)
        self.assertEqual(len(anomalies['quantity_anomalies']), 0)
        
        self.assertIn('time_anomalies', anomalies)
        self.assertEqual(len(anomalies['time_anomalies']), 0)
        
        self.assertIn('format_anomalies', anomalies)
        self.assertEqual(len(anomalies['format_anomalies']), 0)
        
        # Clean data should be empty
        self.assertEqual(len(clean_df), 0)

    def test_insufficient_data_for_iqr(self):
        """Test function behavior with insufficient data points for IQR analysis"""
        # Run the function with small data
        anomalies, clean_df = detect_anomalies(self.small_data)
        
        # Should not detect price or quantity anomalies due to insufficient data
        self.assertIn('price_anomalies', anomalies)
        self.assertEqual(len(anomalies['price_anomalies']), 0)
        
        self.assertIn('quantity_anomalies', anomalies)
        self.assertEqual(len(anomalies['quantity_anomalies']), 0)
        
        # Clean data should match original data (no anomalies removed)
        self.assertEqual(len(clean_df), len(self.small_data))

    def test_error_handling(self):
        """Test error handling in the function"""
        # Create a dataframe with missing required columns
        bad_data = pl.DataFrame({
            'Transaction ID': [1, 2, 3],
            # Missing 'Date' column
            'Product Name': ['Apple', 'Apple', 'Apple'],
            'Unit Price': [10.0, 9.5, 10.5],
            'Quantity': [2, 3, 2]
        })
        
        # Function should raise an exception
        with self.assertRaises(Exception):
            detect_anomalies(bad_data)

    def test_clean_data_integrity(self):
        """Test that the clean data maintains integrity of non-anomalous records"""
        # Run the function
        anomalies, clean_df = detect_anomalies(self.test_data)
        
        # Count all unique anomalous transaction IDs
        all_anomaly_ids = set()
        for anomaly_type in anomalies.values():
            if len(anomaly_type) > 0:
                all_anomaly_ids.update(anomaly_type['Transaction ID'].to_list())
        
        # Verify the clean data contains exactly the records that aren't anomalies
        expected_clean_count = len(self.test_data) - len(all_anomaly_ids)
        self.assertEqual(len(clean_df), expected_clean_count)
        
        # Verify each non-anomalous transaction is present in clean data
        for transaction_id in range(1, 21):
            if transaction_id not in all_anomaly_ids:
                self.assertIn(transaction_id, clean_df['Transaction ID'].to_list())


class TestUploadDfToGcs(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.test_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["John", "Jane", "Bob"],
            "value": [10.5, 20.1, 15.7]
        })
        self.bucket_name = "test-bucket"
        self.blob_name = "test-folder/data.csv"
        
        # Expected CSV content
        self.expected_csv = self.test_df.write_csv()

    @patch('dataPreprocessing.storage')
    @patch('dataPreprocessing.logging')
    def test_successful_upload(self, mock_logging, mock_storage):
        # Configure mocks
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Setup chain of mock returns
        mock_storage.Client.return_value = mock_client
        mock_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Call the function
        upload_df_to_gcs(self.test_df, self.bucket_name, self.blob_name)
        
        # Verify the mocks were called with correct arguments
        mock_client.get_bucket.assert_called_once_with(self.bucket_name)
        mock_bucket.blob.assert_called_once_with(self.blob_name)
        mock_blob.upload_from_string.assert_called_once_with(
            self.expected_csv, content_type='text/csv'
        )
        
        # Verify logging
        mock_logging.info.assert_any_call(
            "Starting upload to GCS. Bucket: %s, Blob: %s", 
            self.bucket_name, self.blob_name
        )
        mock_logging.info.assert_any_call(
            "Upload successful to GCS. Blob name: %s", self.blob_name
        )

    @patch('dataPreprocessing.storage')
    @patch('dataPreprocessing.logging')
    def test_google_cloud_error(self, mock_logging, mock_storage):
        # Configure mocks
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Setup chain of mock returns
        mock_storage.Client.return_value = mock_client
        mock_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Make the upload_from_string method raise a GoogleCloudError
        cloud_error = GoogleCloudError("Storage error")
        mock_blob.upload_from_string.side_effect = cloud_error
        
        # Call the function and check that it raises the error
        with self.assertRaises(GoogleCloudError):
            upload_df_to_gcs(self.test_df, self.bucket_name, self.blob_name)
        
        # Verify logging
        mock_logging.error.assert_called_with(
            "Error uploading DataFrame to GCS. Error: %s", cloud_error
        )

    @patch('dataPreprocessing.storage')
    @patch('dataPreprocessing.logging')
    def test_general_exception(self, mock_logging, mock_storage):
        # Configure mocks
        mock_client = MagicMock()
        
        # Make get_bucket raise a general exception
        general_error = Exception("General error")
        mock_client.get_bucket.side_effect = general_error
        mock_storage.Client.return_value = mock_client
        
        # Call the function and check that it raises the error
        with self.assertRaises(Exception):
            upload_df_to_gcs(self.test_df, self.bucket_name, self.blob_name)
        
        # Verify logging
        mock_logging.error.assert_called_with(
            "Error uploading DataFrame to GCS. Error: %s", general_error
        )

    @patch('dataPreprocessing.storage')
    def test_empty_dataframe(self, mock_storage):
        # Create an empty dataframe
        empty_df = pl.DataFrame()
        
        # Configure mocks
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Setup chain of mock returns
        mock_storage.Client.return_value = mock_client
        mock_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Call the function
        upload_df_to_gcs(empty_df, self.bucket_name, self.blob_name)
        
        # Verify the empty CSV was uploaded
        empty_csv = empty_df.write_csv()
        mock_blob.upload_from_string.assert_called_once_with(
            empty_csv, content_type='text/csv'
        )


if __name__ == '__main__':
    unittest.main()
