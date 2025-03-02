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
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    def test_missing_columns(
        self,
        mock_load_bucket_data,
        mock_list_bucket_blobs,
        mock_delete_blob_from_bucket,
        mock_logger,
        mock_send_email,
        mock_collect_errors,
    ):
        df = pd.DataFrame(
            {
                "Date": ["2023-01-01"],
                "Transaction ID": [123],
                "Quantity": [10],
                "Store Location": ["Store A"],
                "Product Name": ["milk"],
            }
        )

        is_valid, error_message = validate_data(df)

        self.assertFalse(is_valid)
        self.assertIn("Missing columns: Unit Price, Producer ID", error_message)

        mock_collect_errors.assert_called_once_with(
            df, ["Unit Price", "Producer ID"], set(), {}
        )

        mock_send_email.assert_not_called()
        mock_logger.error.assert_called_with(
            "Data validation failed:\nMissing columns: Unit Price, Producer ID"
        )


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
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    def test_exception_handling(
        self,
        mock_load_bucket_data,
        mock_list_bucket_blobs,
        mock_delete_blob_from_bucket,
        mock_logger,
        mock_send_email,
        mock_collect_errors,
    ):
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

        with patch.object(
            df_polars, "to_pandas", side_effect=Exception("Conversion error")
        ):
            is_valid, error_message = validate_data(df_polars)

            self.assertFalse(is_valid)
            self.assertEqual(error_message, "Conversion error")

            mock_send_email.assert_not_called()
            mock_collect_errors.assert_not_called()
            mock_logger.error.assert_called_with("Error in data validation: Conversion error")


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



    # Case 1: Valid file - validation passes.
    
    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    def test_valid_file(
        self,
        mock_logger,
        mock_load_bucket_data,
        mock_validate_data,
        mock_delete_blob,
    ):
        bucket_name = "test_bucket"
        blob_name = "valid_file.csv"

        df = pd.DataFrame({"dummy": [1]})
        mock_load_bucket_data.return_value = df
        mock_validate_data.return_value = (True, None)

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=True)

        self.assertTrue(is_valid)
        self.assertIsNone(error_info)
        mock_delete_blob.assert_not_called()

    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
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
        mock_validate_data.return_value = (False, "Validation error")
        mock_delete_blob.return_value = True

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=True)

        self.assertFalse(is_valid)
        self.assertEqual(error_info["filename"], blob_name)
        self.assertEqual(error_info["error"], "Validation error")
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
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
        mock_validate_data.return_value = (False, "Validation error")
        mock_delete_blob.return_value = False

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=True)

        self.assertFalse(is_valid)
        self.assertEqual(error_info["filename"], blob_name)
        self.assertEqual(error_info["error"], "Validation error")
        self.assertTrue(error_info["deletion_failed"])
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.validate_data")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
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
        mock_validate_data.return_value = (False, "Validation error")

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=False)

        self.assertFalse(is_valid)
        self.assertEqual(error_info["filename"], blob_name)
        self.assertEqual(error_info["error"], "Validation error")
        mock_delete_blob.assert_not_called()

    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    def test_exception_during_validation_delete_success(
        self, mock_logger, mock_load_bucket_data, mock_delete_blob
    ):
        bucket_name = "test_bucket"
        blob_name = "exception_file.csv"

        mock_load_bucket_data.side_effect = Exception("Load error")
        mock_delete_blob.return_value = True

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=True)

        self.assertFalse(is_valid)
        self.assertEqual(error_info["filename"], blob_name)
        self.assertIn("Exception during validation", error_info["error"])
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    def test_exception_during_validation_delete_failure(
        self, mock_logger, mock_load_bucket_data, mock_delete_blob
    ):
        bucket_name = "test_bucket"
        blob_name = "exception_file.csv"

        mock_load_bucket_data.side_effect = Exception("Load error")
        mock_delete_blob.return_value = False

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=True)

        self.assertFalse(is_valid)
        self.assertEqual(error_info["filename"], blob_name)
        self.assertIn("Exception during validation", error_info["error"])
        self.assertTrue(error_info["deletion_failed"])
        mock_delete_blob.assert_called_once_with(bucket_name, blob_name)

    @patch("Data_Pipeline.scripts.pre_validation.delete_blob_from_bucket")
    @patch("Data_Pipeline.scripts.pre_validation.load_bucket_data")
    @patch("Data_Pipeline.scripts.pre_validation.logger")
    def test_exception_during_validation_no_deletion(
        self, mock_logger, mock_load_bucket_data, mock_delete_blob
    ):
        bucket_name = "test_bucket"
        blob_name = "exception_file.csv"

        mock_load_bucket_data.side_effect = Exception("Load error")

        is_valid, error_info = validate_file(bucket_name, blob_name, delete_invalid=False)

        self.assertFalse(is_valid)
        self.assertEqual(error_info["filename"], blob_name)
        self.assertIn("Exception during validation", error_info["error"])
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
    @patch("Data_Pipeline.scripts.pre_validation.collect_validation_errors")
    def test_empty_dataframe_validate_data(self, mock_collect_validation_errors, mock_logger):
        df_empty = pd.DataFrame()

        is_valid, error_message = validate_data(df_empty)

        self.assertFalse(is_valid)
        self.assertEqual(error_message, "DataFrame is empty, no rows found")
        mock_logger.error.assert_called_with("Data validation failed:\nDataFrame is empty, no rows found")
        mock_collect_validation_errors.assert_not_called()


    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    def test_main_all_files_invalid_main(
        self, mock_send_email, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        bucket_name = "test-bucket"
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names
        mock_validate_file.side_effect = lambda *args, **kwargs: (False, {"filename": args[1], "error": "Validation failed"})

        ret = main(bucket_name=bucket_name)

        self.assertEqual(ret, 2)
        mock_send_email.assert_called_once()
        mock_logger.error.assert_called_with("All files failed validation.")



    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    def test_main_cloud_all_valid(
        self, mock_send_email, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        bucket_name = "test-bucket"
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names
        mock_validate_file.side_effect = lambda *args, **kwargs: (True, None)

        ret = main(bucket_name=bucket_name)

        self.assertEqual(ret, 0)
        mock_send_email.assert_not_called()

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    def test_main_cloud_partial_valid(
        self, mock_send_email, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        bucket_name = "test-bucket"
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names

        def side_effect(bucket, blob, delete_invalid):
            return (blob != "file2.csv", {"filename": blob, "error": "Validation failed"} if blob == "file2.csv" else None)

        mock_validate_file.side_effect = side_effect

        ret = main(bucket_name=bucket_name)

        self.assertEqual(ret, 1)
        mock_send_email.assert_called_once()

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    def test_main_cloud_all_invalid(
        self, mock_send_email, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        bucket_name = "test-bucket"
        blob_names = ["file1.csv", "file2.csv", "file3.csv"]
        mock_list_bucket_blobs.return_value = blob_names
        mock_validate_file.side_effect = lambda *args, **kwargs: (False, {"filename": args[1], "error": "Validation failed"})

        ret = main(bucket_name=bucket_name)

        self.assertEqual(ret, 2)
        mock_send_email.assert_called_once()

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    def test_main_no_files_in_bucket(
        self, mock_send_email, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        bucket_name = "test-bucket"
        mock_list_bucket_blobs.return_value = []

        ret = main(bucket_name=bucket_name)

        self.assertEqual(ret, 2)
        mock_logger.warning.assert_called_with("No files found in bucket 'test-bucket'")
        mock_send_email.assert_not_called()

    @patch("Data_Pipeline.scripts.pre_validation.logger")
    @patch("Data_Pipeline.scripts.pre_validation.list_bucket_blobs")
    @patch("Data_Pipeline.scripts.pre_validation.validate_file")
    @patch("Data_Pipeline.scripts.pre_validation.send_email")
    def test_main_exception_handling(
        self, mock_send_email, mock_validate_file, mock_list_bucket_blobs, mock_logger
    ):
        bucket_name = "test-bucket"
        mock_list_bucket_blobs.side_effect = Exception("Unexpected error")

        ret = main(bucket_name=bucket_name)

        self.assertEqual(ret, 2)
        mock_logger.error.assert_called()
        mock_send_email.assert_called_once()


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
