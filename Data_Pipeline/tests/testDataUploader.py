import pandas as pd
from polars.testing import assert_frame_equal
from datetime import datetime
import tempfile
import io
import unittest
from google.api_core.exceptions import Conflict
from unittest.mock import MagicMock, patch
from scripts.dataUploader import upload_json_to_gcs, excel_to_json, process_all_excel_files_in_data_folder

class TestDataUploader(unittest.TestCase):


    ### Unit Tests for upload_json_to_gcs function.

    # Test case where bucket exist.
    @patch("scripts.dataPreprocessing.storage.Client")
    def test_upload_json_to_gcs_bucket_exist(self, mock_client_cls):
        # Setup
        bucket_mock = MagicMock()
        blob_mock = MagicMock()
        bucket_mock.blob.return_value = blob_mock

        client_instance = MagicMock()
        client_instance.get_bucket.return_value = bucket_mock
        mock_client_cls.return_value = client_instance

        bucket_name = "my-bucket"
        json_data = '[{"Date": "2024-01-01T00:00:00", "Unit Price": 100.5, "Quantity": 10, "Transaction ID": "T123", "Store Location": "NY", "Product Name": "milk", "Producer ID": "P001"}, {"Date": "2024-01-02T00:00:00", "Unit Price": 200.75, "Quantity": 20, "Transaction ID": "T456", "Store Location": "CA", "Product Name": "coffee", "Producer ID": "P002"}]'
        destination_blob_name = "data.json"

        # Test
        upload_json_to_gcs(bucket_name, json_data, destination_blob_name)        

        # Assert
        client_instance.get_bucket.assert_called_with(bucket_name)
        bucket_mock.blob.assert_called_with(destination_blob_name)
        blob_mock.upload_from_string.assert_called_with(
            json_data, content_type="application/json"
        )


    # Test case where bucket not exist and created.
    @patch("scripts.dataPreprocessing.storage.Client")
    def test_upload_json_to_gcs_bucket_not_exist_and_created(self, mock_client_cls):
        # Setup
        client_instance = MagicMock()
        client_instance.get_bucket.side_effect = Exception("Bucket Not Found!")

        bucket_mock = MagicMock()
        blob_mock = MagicMock()
        bucket_mock.blob.return_value = blob_mock
        client_instance.create_bucket.return_value = bucket_mock

        mock_client_cls.return_value = client_instance

        bucket_name = "my-bucket"
        json_data = '[{"Date": "2024-01-01T00:00:00", "Unit Price": 100.5, "Quantity": 10, "Transaction ID": "T123", "Store Location": "NY", "Product Name": "milk", "Producer ID": "P001"}, {"Date": "2024-01-02T00:00:00", "Unit Price": 200.75, "Quantity": 20, "Transaction ID": "T456", "Store Location": "CA", "Product Name": "coffee", "Producer ID": "P002"}]'
        destination_blob_name = "data.json"

        # Test
        upload_json_to_gcs(bucket_name, json_data, destination_blob_name)

        # Assert
        client_instance.get_bucket.assert_called_with(bucket_name)
        client_instance.create_bucket.assert_called_with(bucket_name)
        bucket_mock.blob.assert_called_with(destination_blob_name)
        blob_mock.upload_from_string.assert_called_with(
            json_data, content_type="application/json"
        )

    
    # Test case where bucket creation conflict.
    @patch("scripts.dataPreprocessing.storage.Client")
    def test_upload_json_to_gcs_bucket_bucket_creation_conflict(self, mock_client_cls):
        # Setup
        client_instance = MagicMock()
        client_instance.get_bucket.side_effect = Exception("Bucket Not Found!")
        client_instance.create_bucket.side_effect = Conflict("Bucket conflict")

        bucket_mock = MagicMock()
        blob_mock = MagicMock()
        bucket_mock.blob.return_value = blob_mock
        client_instance.get_bucket.side_effect = [Exception("Bucket not found"), bucket_mock]
        mock_client_cls.return_value = client_instance

        bucket_name = "my-bucket"
        json_data = '[{"Date": "2024-01-01T00:00:00", "Unit Price": 100.5, "Quantity": 10, "Transaction ID": "T123", "Store Location": "NY", "Product Name": "milk", "Producer ID": "P001"}, {"Date": "2024-01-02T00:00:00", "Unit Price": 200.75, "Quantity": 20, "Transaction ID": "T456", "Store Location": "CA", "Product Name": "coffee", "Producer ID": "P002"}]'
        destination_blob_name = "data.json"

        # Test
        upload_json_to_gcs(bucket_name, json_data, destination_blob_name)

        # Assert
        self.assertTrue(client_instance.create_bucket.called)
        self.assertEqual(client_instance.get_bucket.call_count, 2)
        bucket_mock.blob.assert_called_with(destination_blob_name)
        blob_mock.upload_from_string.assert_called_with(
            json_data, content_type="application/json"
        )


    # Test case where upload fails.
    @patch("scripts.dataPreprocessing.storage.Client")
    def test_upload_json_to_gcs_upload_fails(self, mock_client_cls):
        # Setup
        bucket_mock = MagicMock()
        blob_mock = MagicMock()
        blob_mock.upload_from_string.side_effect = Exception("Upload failed")
        bucket_mock.blob.return_value = blob_mock

        client_instance = MagicMock()
        client_instance.get_bucket.return_value = bucket_mock
        mock_client_cls.return_value = client_instance

        bucket_name = "my-bucket"
        json_data = '[{"Date": "2024-01-01T00:00:00", "Unit Price": 100.5, "Quantity": 10, "Transaction ID": "T123", "Store Location": "NY", "Product Name": "milk", "Producer ID": "P001"}, {"Date": "2024-01-02T00:00:00", "Unit Price": 200.75, "Quantity": 20, "Transaction ID": "T456", "Store Location": "CA", "Product Name": "coffee", "Producer ID": "P002"}]'
        destination_blob_name = "data.json"

        # Test
        with self.assertRaises(Exception) as context:
            upload_json_to_gcs(bucket_name, json_data, destination_blob_name)

        # Assert
        self.assertIn("Upload failed", str(context.exception))


    

    ### Unit Tests for process_all_excel_files_in_data_folder function.

    # Test case where excel file exist.
    # Test case where no excel file exist.
    # Test case where Exception in excel_to_json.
    # Test case where Exception in upload_json_to_gcs.