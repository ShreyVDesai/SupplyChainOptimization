import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from Data_Pipeline.scripts.pre_validation import *


class TestPreValidateData(unittest.TestCase):

    # We patch PRE_VALIDATION_COLUMNS to a known list for predictable tests.
    @patch(
        "Data_Pipeline.scripts.pre_validation.PRE_VALIDATION_COLUMNS",
        [
            "Date",
            "Unit Price",
            "Transaction ID",
            "Quantity",
            "Producer ID",
            "Store Location",
            "Product Name",
        ],
    )
    @patch("Data_Pipeline.scripts.pre_validation.collect_validation_errors")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    @patch("Data_Pipeline.scripts.logger")
    def test_all_columns_present_pandas(
        self, mock_logger, mock_send_email, mock_collect_errors
    ):
        """All required columns are present (pandas DataFrame) → returns True."""
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Unit Price": [2.5],
                "Transaction ID": [123],
                "Quantity": [10],
                "Producer ID": [456],
                "Store Location": ["Store A"],
                "Product Name": ["milk"],
            }
        )

        result = validate_data(df)
        self.assertTrue(result)
        mock_send_email.assert_not_called()
        mock_collect_errors.assert_not_called()

    # We patch PRE_VALIDATION_COLUMNS to a known list for predictable tests.
    @patch(
        "Data_Pipeline.scripts.pre_validation.PRE_VALIDATION_COLUMNS",
        [
            "Date",
            "Unit Price",
            "Transaction ID",
            "Quantity",
            "Producer ID",
            "Store Location",
            "Product Name",
        ],
    )
    @patch("Data_Pipeline.scripts.pre_validation.collect_validation_errors")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    @patch("Data_Pipeline.scripts.logger")
    def test_missing_columns(
        self, mock_logger, mock_send_email, mock_collect_errors
    ):
        """
        When one or more required columns are missing (pandas DataFrame),
        validate_data should return False, call collect_validation_errors and send_email,
        and log an error.
        """
        # Create DataFrame missing "Unit Price" and "Producer ID"
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                # "Unit Price" is missing
                "Transaction ID": [123],
                "Quantity": [10],
                # "Producer ID" is missing
                "Store Location": ["Store A"],
                "Product Name": ["milk"],
            }
        )

        result = validate_data(df)
        self.assertFalse(result)
        mock_collect_errors.assert_called_once()  # Errors should be collected.
        mock_send_email.assert_called_once()  # An email should be sent.

    # We patch PRE_VALIDATION_COLUMNS to a known list for predictable tests.
    @patch(
        "Data_Pipeline.scripts.pre_validation.PRE_VALIDATION_COLUMNS",
        [
            "Date",
            "Unit Price",
            "Transaction ID",
            "Quantity",
            "Producer ID",
            "Store Location",
            "Product Name",
        ],
    )
    @patch("Data_Pipeline.scripts.pre_validation.collect_validation_errors")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    @patch("Data_Pipeline.scripts.logger")
    def test_polars_input_all_columns(
        self, mock_logger, mock_send_email, mock_collect_errors
    ):
        """
        When a Polars DataFrame with all required columns is provided,
        it is converted to pandas and validation passes.
        """
        df_polars = pl.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Unit Price": [2.5],
                "Transaction ID": [123],
                "Quantity": [10],
                "Producer ID": [456],
                "Store Location": ["Store A"],
                "Product Name": ["milk"],
            }
        )
        result = validate_data(df_polars)
        self.assertTrue(result)
        mock_send_email.assert_not_called()
        mock_collect_errors.assert_not_called()

    # We patch PRE_VALIDATION_COLUMNS to a known list for predictable tests.
    @patch(
        "Data_Pipeline.scripts.pre_validation.PRE_VALIDATION_COLUMNS",
        [
            "Date",
            "Unit Price",
            "Transaction ID",
            "Quantity",
            "Producer ID",
            "Store Location",
            "Product Name",
        ],
    )
    @patch("Data_Pipeline.scripts.pre_validation.collect_validation_errors")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    @patch("Data_Pipeline.scripts.logger")
    def test_exception_handling(
        self, mock_logger, mock_send_email, mock_collect_errors
    ):
        """
        If an exception occurs during conversion (e.g. df.to_pandas() raises an error),
        validate_data should catch the exception, log an error, and return False.
        """
        df_polars = pl.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Unit Price": [2.5],
                "Transaction ID": [123],
                "Quantity": [10],
                "Producer ID": [456],
                "Store Location": ["Store A"],
                "Product Name": ["milk"],
            }
        )
        # Patch the to_pandas method to raise an exception.
        with patch.object(
            df_polars, "to_pandas", side_effect=Exception("Conversion error")
        ):
            result = validate_data(df_polars)
            self.assertFalse(result)
            mock_send_email.assert_not_called()
            mock_collect_errors.assert_not_called()

    def test_no_missing_columns_collect_validation_errors(self):
        """When missing_columns is empty, error indices and reasons remain unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        error_indices = set()
        error_reasons = {}
        collect_validation_errors(df, [], error_indices, error_reasons)
        self.assertEqual(error_indices, set())
        self.assertEqual(error_reasons, {})

    def test_missing_columns_non_empty_pandas_collect_validation_errors(self):
        """
        For a pandas DataFrame with missing columns,
        all row indices should be added and error reasons populated.
        """
        df = pd.DataFrame({"a": [1, 2, 3]})
        missing_columns = ["col1", "col2"]
        error_indices = set()
        error_reasons = {}
        collect_validation_errors(
            df, missing_columns, error_indices, error_reasons
        )
        expected_indices = {0, 1, 2}
        self.assertEqual(error_indices, expected_indices)
        expected_message = f"Missing columns: {', '.join(missing_columns)}"
        for idx in range(len(df)):
            self.assertEqual(error_reasons[idx], [expected_message])

    def test_empty_dataframe_collect_validation_errors(self):
        """
        For an empty DataFrame (pandas), even with missing columns,
        no error indices or reasons should be added.
        """
        df = pd.DataFrame({"a": []})
        missing_columns = ["col1", "col2"]
        error_indices = set()
        error_reasons = {}
        collect_validation_errors(
            df, missing_columns, error_indices, error_reasons
        )
        self.assertEqual(error_indices, set())
        self.assertEqual(error_reasons, {})

    def test_polars_dataframe_missing_columns_collect_validation_errors(self):
        """
        For a Polars DataFrame with missing columns,
        error_indices should include all row indices and error_reasons set accordingly.
        """
        df = pl.DataFrame({"a": [10, 20, 30, 40]})
        missing_columns = ["colX"]
        error_indices = set()
        error_reasons = {}
        collect_validation_errors(
            df, missing_columns, error_indices, error_reasons
        )
        expected_indices = {0, 1, 2, 3}
        self.assertEqual(error_indices, expected_indices)
        expected_message = f"Missing columns: {', '.join(missing_columns)}"
        for idx in range(len(df)):
            self.assertEqual(error_reasons[idx], [expected_message])

    @patch("Data_Pipeline.scripts.pre_validation.setup_gcp_credentials")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.storage.Client")
    def test_list_bucket_blobs_success(
        self, mock_storage_client, mock_logger, mock_setup_creds
    ):
        # Create dummy blob objects with a 'name' attribute.
        dummy_blob1 = MagicMock()
        dummy_blob1.name = "file1.txt"
        dummy_blob2 = MagicMock()
        dummy_blob2.name = "file2.txt"
        dummy_blobs = [dummy_blob1, dummy_blob2]

        # Set up a dummy bucket that returns our list of blobs.
        dummy_bucket = MagicMock()
        dummy_bucket.list_blobs.return_value = dummy_blobs

        # Create a dummy storage client that returns our dummy bucket.
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        bucket_name = "test_bucket"
        result = list_bucket_blobs(bucket_name)

        # Verify that setup_gcp_credentials was called.
        mock_setup_creds.assert_called_once()

        # Verify that the storage client was used correctly.
        dummy_storage_instance.get_bucket.assert_called_once_with(bucket_name)
        dummy_bucket.list_blobs.assert_called_once()

        # Check that logger.info was called with a message that includes the
        # correct file count.
        expected_info = (
            f"Found {len(dummy_blobs)} files in bucket '{bucket_name}'"
        )
        mock_logger.info.assert_any_call(expected_info)

        # Verify that the function returns the correct list of blob names.
        self.assertEqual(result, ["file1.txt", "file2.txt"])

    @patch("Data_Pipeline.scripts.pre_validation.setup_gcp_credentials")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.storage.Client")
    def test_list_bucket_blobs_exception(
        self, mock_storage_client, mock_logger, mock_setup_creds
    ):
        bucket_name = "test_bucket"

        # Simulate an exception when get_bucket is called.
        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.side_effect = Exception(
            "Bucket error"
        )
        mock_storage_client.return_value = dummy_storage_instance

        with self.assertRaises(Exception) as context:
            list_bucket_blobs(bucket_name)
        self.assertIn("Bucket error", str(context.exception))

        # Verify that setup_gcp_credentials was called.
        mock_setup_creds.assert_called_once()

        # Check that logger.error was called with an error message that
        # includes "Bucket error"
        error_calls = [
            str(args[0]) for args, _ in mock_logger.error.call_args_list
        ]
        self.assertTrue(
            any("Bucket error" in msg for msg in error_calls),
            "Expected error log containing 'Bucket error'.",
        )

    @patch("Data_Pipeline.scripts.pre_validation.setup_gcp_credentials")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.storage.Client")
    def test_delete_blob_success(
        self, mock_storage_client, mock_logger, mock_setup_creds
    ):
        # Set up a dummy bucket and blob
        dummy_blob = MagicMock()
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob

        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        bucket_name = "test_bucket"
        blob_name = "test_blob.txt"

        # Call the function
        result = delete_blob_from_bucket(bucket_name, blob_name)

        # Verify that setup_gcp_credentials was called.
        mock_setup_creds.assert_called_once()

        # Verify that the storage client got the correct bucket and blob.
        dummy_storage_instance.get_bucket.assert_called_once_with(bucket_name)
        dummy_bucket.blob.assert_called_once_with(blob_name)

        # Verify that blob.delete was called.
        dummy_blob.delete.assert_called_once()

        # Check that the success log message was recorded.
        expected_log_msg = (
            f"Blob {blob_name} deleted from bucket {bucket_name}"
        )
        mock_logger.info.assert_any_call(expected_log_msg)

        # The function should return True.
        self.assertTrue(result)

    @patch("Data_Pipeline.scripts.pre_validation.setup_gcp_credentials")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.storage.Client")
    def test_delete_blob_failure_delete_blob(
        self, mock_storage_client, mock_logger, mock_setup_creds
    ):
        # Set up a dummy bucket and blob that will raise an exception on
        # delete.
        dummy_blob = MagicMock()
        dummy_blob.delete.side_effect = Exception("Deletion failed")
        dummy_bucket = MagicMock()
        dummy_bucket.blob.return_value = dummy_blob

        dummy_storage_instance = MagicMock()
        dummy_storage_instance.get_bucket.return_value = dummy_bucket
        mock_storage_client.return_value = dummy_storage_instance

        bucket_name = "test_bucket"
        blob_name = "test_blob.txt"

        # Call the function; it should catch the exception and return False.
        result = delete_blob_from_bucket(bucket_name, blob_name)

        # Verify that setup_gcp_credentials was called.
        mock_setup_creds.assert_called_once()

        # Verify that get_bucket and blob methods were called.
        dummy_storage_instance.get_bucket.assert_called_once_with(bucket_name)
        dummy_bucket.blob.assert_called_once_with(blob_name)
        dummy_blob.delete.assert_called_once()

        # Check that an error was logged.
        error_calls = [
            str(arg)
            for args, _ in mock_logger.error.call_args_list
            for arg in args
        ]
        self.assertTrue(
            any("Deletion failed" in msg for msg in error_calls),
            "Expected error log containing 'Deletion failed'.",
        )

        # The function should return False.
        self.assertFalse(result)

    # Case 1: Valid file - validation passes.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_valid_file(
        self,
        mock_logger,
        mock_load_bucket_data,
        mock_validate_data,
        mock_delete_blob,
    ):
        bucket_name = "test_bucket"
        blob_name = "valid_file.csv"

        # Simulate load_bucket_data returns a DataFrame (could be pandas or
        # polars).
        df = pd.DataFrame({"dummy": [1]})
        mock_load_bucket_data.return_value = df
        # Simulate validation passes.
        mock_validate_data.return_value = True

        result = validate_file(bucket_name, blob_name, delete_invalid=True)
        self.assertTrue(result)
        mock_delete_blob.assert_not_called()

    # Case 2: Invalid file with deletion enabled and deletion succeeds.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_invalid_file_delete_success(
        self,
        mock_logger,
        mock_load_bucket_data,
        mock_validate_data,
        mock_delete_blob,
    ):
        bucket_name = "test_bucket"
        blob_name = "invalid_file.csv"

        df = pd.DataFrame({"dummy": [1]})
        mock_load_bucket_data.return_value = df
        # Simulate validation fails.
        mock_validate_data.return_value = False
        # Simulate deletion returns True.
        mock_delete_blob.return_value = True

        result = validate_file(bucket_name, blob_name, delete_invalid=True)
        self.assertFalse(result)
        # And delete_blob_from_bucket was called.
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    # Case 3: Invalid file with deletion enabled and deletion fails.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_invalid_file_delete_failure(
        self,
        mock_logger,
        mock_load_bucket_data,
        mock_validate_data,
        mock_delete_blob,
    ):
        bucket_name = "test_bucket"
        blob_name = "invalid_file.csv"

        df = pd.DataFrame({"dummy": [1]})
        mock_load_bucket_data.return_value = df
        mock_validate_data.return_value = False
        # Simulate deletion returns False.
        mock_delete_blob.return_value = False

        result = validate_file(bucket_name, blob_name, delete_invalid=True)
        self.assertFalse(result)
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    # Case 4: Invalid file with deletion disabled.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_invalid_file_no_deletion(
        self,
        mock_logger,
        mock_load_bucket_data,
        mock_validate_data,
        mock_delete_blob,
    ):
        bucket_name = "test_bucket"
        blob_name = "invalid_file.csv"

        df = pd.DataFrame({"dummy": [1]})
        mock_load_bucket_data.return_value = df
        mock_validate_data.return_value = False

        result = validate_file(bucket_name, blob_name, delete_invalid=False)
        self.assertFalse(result)
        mock_delete_blob.assert_not_called()

    # Case 5: Exception occurs during validation with deletion enabled and
    # deletion succeeds.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_exception_during_validation_delete_success(
        self, mock_logger, mock_load_bucket_data, mock_delete_blob
    ):
        bucket_name = "test_bucket"
        blob_name = "exception_file.csv"

        # Simulate load_bucket_data raises an exception.
        mock_load_bucket_data.side_effect = Exception("Load error")
        # Simulate deletion returns True.
        mock_delete_blob.return_value = True

        result = validate_file(bucket_name, blob_name, delete_invalid=True)
        self.assertFalse(result)
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    # Case 6: Exception occurs during validation with deletion enabled and
    # deletion fails.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_exception_during_validation_delete_failure(
        self, mock_logger, mock_load_bucket_data, mock_delete_blob
    ):
        bucket_name = "test_bucket"
        blob_name = "exception_file.csv"

        mock_load_bucket_data.side_effect = Exception("Load error")
        # Simulate deletion returns False.
        mock_delete_blob.return_value = False

        result = validate_file(bucket_name, blob_name, delete_invalid=True)
        self.assertFalse(result)
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    # Case 7: Exception occurs during validation with deletion disabled.
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.logger")
    def test_exception_during_validation_no_deletion(
        self, mock_logger, mock_load_bucket_data, mock_delete_blob
    ):
        bucket_name = "test_bucket"
        blob_name = "exception_file.csv"

        mock_load_bucket_data.side_effect = Exception("Load error")

        result = validate_file(bucket_name, blob_name, delete_invalid=False)
        self.assertFalse(result)
        mock_delete_blob.assert_not_called()

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    def test_main_cloud_no_files(
        self, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        # Simulate no files found.
        mock_list_bucket_blobs.return_value = []
        bucket_name = "test-bucket"
        ret = main(bucket_name=bucket_name)
        self.assertEqual(ret, 2)
        mock_list_bucket_blobs.assert_called_once_with(bucket_name)

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    def test_main_cloud_all_valid(
        self, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        # Simulate three files; all validate successfully.
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names
        mock_validate_file.return_value = True
        bucket_name = "test-bucket"
        ret = main(bucket_name=bucket_name)
        self.assertEqual(ret, 0)

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    def test_main_cloud_partial_valid(
        self, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        # Simulate three files; two valid, one invalid.
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names

        def side_effect(bucket, blob, delete_invalid):
            return blob != "file2.csv"  # Return False for file2.csv

        mock_validate_file.side_effect = side_effect
        bucket_name = "test-bucket"
        ret = main(bucket_name=bucket_name)
        self.assertEqual(ret, 1)

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    def test_main_cloud_all_invalid(
        self, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        # Simulate three files; all invalid.
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names
        mock_validate_file.return_value = False
        bucket_name = "test-bucket"
        ret = main(bucket_name=bucket_name)
        self.assertEqual(ret, 2)

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    def test_main_cloud_exception(self, mock_list_bucket_blobs, mock_logger):
        # Simulate exception during file listing.
        mock_list_bucket_blobs.side_effect = Exception("List error")
        bucket_name = "test-bucket"
        ret = main(bucket_name=bucket_name)
        self.assertEqual(ret, 2)


if __name__ == "__main__":
    unittest.main()
