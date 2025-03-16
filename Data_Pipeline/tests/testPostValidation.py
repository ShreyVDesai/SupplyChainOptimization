import unittest
import pandas as pd
import polars as pl
from scripts.post_validation import check_column_types, validate_data, generate_numeric_stats, main
from unittest.mock import MagicMock, call, patch

class TestPostiValidation(unittest.TestCase):
    def setUp(self):
        self.error_indices = set()
        self.error_reasons = {}

    
    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.pd.to_numeric", side_effect=Exception("Test conversion error"))
    def test_total_quantity_exception(self, mock_to_numeric, mock_logger):
        # Create a sample DataFrame with the "Total Quantity" column.
        df = pd.DataFrame({
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [10, 20],
            "Date": ["2023-01-01", "2023-01-02"]
        })
        error_indices = set()
        error_reasons = {}

        # Call check_column_types, which will trigger the exception
        check_column_types(df, error_indices, error_reasons)

        # Verify that logger.error was called with the expected error message.
        mock_logger.error.assert_called_with("Error validating Total Quantity: Test conversion error")


    def test_date_exception(self):
        # Define a custom string that raises an exception when len() is called.
        class BadStr(str):
            def __len__(self):
                raise ValueError("Bad length")
        
        # Create a DataFrame with one row where the Date column is a BadStr.
        df = pd.DataFrame({
            "Product Name": ["Apple"],
            "Total Quantity": [10],
            "Date": [BadStr("dummy")]
        })
        
        error_indices = set()
        error_reasons = {}
        
        # Call check_column_types, which should trigger the exception block for the date value.
        check_column_types(df, error_indices, error_reasons)
        
        # Verify that the error was recorded for row index 0.
        self.assertIn(0, error_indices)
        # Verify that the error message includes "Invalid date format" and "Bad length"
        self.assertTrue(
            any("Bad length" in reason for reason in error_reasons.get(0, [])),
            f"Expected an error message containing 'Bad length', got: {error_reasons.get(0)}"
        )


    def test_valid_data(self):
        df = pd.DataFrame({
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [10, 5],
            "Date": ["2023-01-01", "01/02/2023"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertEqual(len(self.error_indices), 0)
        self.assertEqual(len(self.error_reasons), 0)

    def test_invalid_product_name(self):
        df = pd.DataFrame({
            "Product Name": [123, None, "Milk"],
            "Total Quantity": [10, 5, 7],
            "Date": ["2023-01-01", "01/02/2023", "02-03-2023"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertIn(0, self.error_indices)
        self.assertIn(1, self.error_indices)
        self.assertIn("Product Name must be a string", self.error_reasons[0])
        self.assertIn("Product Name must be a string", self.error_reasons[1])

    def test_total_quantity_not_numeric(self):
        df = pd.DataFrame({
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": ["ten", None],
            "Date": ["2023-01-01", "01/02/2023"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertIn(0, self.error_indices)
        self.assertIn(1, self.error_indices)
        self.assertIn("Total Quantity must be numeric", self.error_reasons[0])
        self.assertIn("Total Quantity must be numeric", self.error_reasons[1])

    def test_total_quantity_negative_or_zero(self):
        df = pd.DataFrame({
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [-5, 0],
            "Date": ["2023-01-01", "01/02/2023"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertIn(0, self.error_indices)
        self.assertIn(1, self.error_indices)
        self.assertIn("Total Quantity must be greater than 0", self.error_reasons[0])
        self.assertIn("Total Quantity must be greater than 0", self.error_reasons[1])

    def test_invalid_date_format(self):
        df = pd.DataFrame({
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [10, 5],
            "Date": ["2023/01/01", "ABC"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertIn(1, self.error_indices)
        self.assertIn("Date must be in a valid date format", self.error_reasons[1])

    def test_date_null(self):
        df = pd.DataFrame({
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [10, 5],
            "Date": [None, "2023-01-01"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertIn(0, self.error_indices)
        self.assertIn("Date cannot be null", self.error_reasons[0])

    
    def test_multiple_errors(self):
        df = pd.DataFrame({
            "Product Name": [None, 123, "Milk"],
            "Total Quantity": ["invalid", -5, 10],
            "Date": ["2023/01/01", None, "Invalid"]
        })
        check_column_types(df, self.error_indices, self.error_reasons)
        self.assertIn(0, self.error_indices)
        self.assertIn(1, self.error_indices)
        self.assertIn(2, self.error_indices)

        self.assertIn("Product Name must be a string", self.error_reasons[0])
        self.assertIn("Total Quantity must be numeric", self.error_reasons[0])
        self.assertIn("Date must be in a valid date format", self.error_reasons[0])

        self.assertIn("Product Name must be a string", self.error_reasons[1])
        self.assertIn("Total Quantity must be greater than 0", self.error_reasons[1])
        self.assertIn("Date cannot be null", self.error_reasons[1])

        self.assertIn("Date must be in a valid date format", self.error_reasons[2])


    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.send_anomaly_alert")
    @patch("scripts.post_validation.collect_validation_errors")
    @patch("scripts.post_validation.check_column_types")
    def test_valid_data_validate_data_post(
        self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    ):
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": [10],
        })
        # Let check_column_types run without adding errors.
        mock_check_column_types.return_value = None

        result = validate_data(df)

        self.assertFalse(result["has_errors"])
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["missing_columns"], [])
        mock_send_anomaly_alert.assert_not_called()

    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.send_anomaly_alert")
    @patch("scripts.post_validation.collect_validation_errors")
    @patch("scripts.post_validation.check_column_types")
    def test_missing_columns(
        self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    ):
        # DataFrame missing "Total Quantity"
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
        })
        
        # Simulate that collect_validation_errors adds an error for the missing column
        def fake_collect(df, missing_cols, error_indices, error_reasons):
            error_indices.add(0)
            error_reasons[0] = ["Missing value for Total Quantity"]
        mock_collect_validation_errors.side_effect = fake_collect

        result = validate_data(df)

        self.assertTrue(result["has_errors"])
        self.assertGreater(result["error_count"], 0)
        self.assertIn("Total Quantity", result["missing_columns"])
        mock_collect_validation_errors.assert_called()
        mock_send_anomaly_alert.assert_called()

    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.send_anomaly_alert")
    @patch("scripts.post_validation.collect_validation_errors")
    @patch("scripts.post_validation.check_column_types")
    def test_anomalies_detected(
        self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    ):
        # DataFrame with a non-numeric value for Total Quantity to trigger an anomaly
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": ["invalid"],
        })

        def fake_check(df, error_indices, error_reasons):
            error_indices.add(0)
            error_reasons[0] = ["Total Quantity must be numeric"]
        mock_check_column_types.side_effect = fake_check

        result = validate_data(df)

        self.assertTrue(result["has_errors"])
        self.assertEqual(result["error_count"], 1)
        mock_send_anomaly_alert.assert_called()

    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.send_anomaly_alert")
    @patch("scripts.post_validation.collect_validation_errors")
    @patch("scripts.post_validation.check_column_types")
    def test_polars_dataframe(
        self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    ):
        # Polars DataFrame should be converted to Pandas and validated normally.
        df = pl.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": [10],
        })

        result = validate_data(df)

        self.assertFalse(result["has_errors"])
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["missing_columns"], [])
        mock_send_anomaly_alert.assert_not_called()



    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.send_anomaly_alert")
    @patch("scripts.post_validation.collect_validation_errors")
    @patch("scripts.post_validation.check_column_types")
    def test_exception_handling(
        self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    ):
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": [10],
        })

        mock_check_column_types.side_effect = Exception("Unexpected error")

        with self.assertRaises(Exception):
            validate_data(df)


    @patch("scripts.post_validation.upload_to_gcs")
    @patch("scripts.post_validation.logger")
    def test_valid_pandas_dataframe(self, mock_logger, mock_upload):
        # Create a sample Pandas DataFrame with two products.
        data = {
            "Product Name": ["Apple", "Apple", "Banana", "Banana"],
            "Total Quantity": [10, 20, 5, 15],
            "Unit Price": [1.0, 1.2, 0.8, 0.9],
        }
        df = pd.DataFrame(data)
        filename = "stats"  # no extension; should become "stats.json"
        
        grouped_stats = generate_numeric_stats(df, filename)
        
        # Check that grouped_stats has keys for each product.
        self.assertIn("Apple", grouped_stats)
        self.assertIn("Banana", grouped_stats)
        # For each product, check that stats for both columns are computed.
        self.assertIn("Total Quantity", grouped_stats["Apple"])
        self.assertIn("Unit Price", grouped_stats["Apple"])
        
        # Verify that upload_to_gcs was called with the filename corrected to "stats.json".
        mock_upload.assert_called_once()
        args, kwargs = mock_upload.call_args
        self.assertEqual(kwargs["bucket_name"], "metadata_stats")
        self.assertEqual(kwargs["destination_blob_name"], "stats.json")
        
        # Verify logger.info was called with a success message.
        mock_logger.info.assert_called_with("Numeric statistics saved to stats.json.")

    @patch("scripts.post_validation.upload_to_gcs")
    @patch("scripts.post_validation.logger")
    def test_valid_polars_dataframe(self, mock_logger, mock_upload):
        # Create a sample Polars DataFrame.
        data = {
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [10, 5],
            "Unit Price": [1.0, 0.8],
        }
        df = pl.DataFrame(data)
        filename = "numeric_stats.json"  # already has extension
        
        grouped_stats = generate_numeric_stats(df, filename)
        
        # Check that grouped_stats contains the products.
        self.assertIn("Apple", grouped_stats)
        self.assertIn("Banana", grouped_stats)
        # Verify filename remains unchanged.
        mock_upload.assert_called_once()
        args, kwargs = mock_upload.call_args
        self.assertEqual(kwargs["destination_blob_name"], "numeric_stats.json")
        mock_logger.info.assert_called_with("Numeric statistics saved to numeric_stats.json.")

    @patch("scripts.post_validation.upload_to_gcs")
    @patch("scripts.post_validation.logger")
    def test_missing_include_column(self, mock_logger, mock_upload):
        # Create a DataFrame missing the "Unit Price" column.
        data = {
            "Product Name": ["Apple", "Banana"],
            "Total Quantity": [10, 5],
        }
        df = pd.DataFrame(data)
        filename = "stats"
        
        grouped_stats = generate_numeric_stats(df, filename)
        
        # For each product, only "Total Quantity" should be computed.
        for product in grouped_stats:
            self.assertIn("Total Quantity", grouped_stats[product])
            self.assertNotIn("Unit Price", grouped_stats[product])

    @patch("scripts.post_validation.upload_to_gcs")
    @patch("scripts.post_validation.logger")
    def test_empty_dataframe(self, mock_logger, mock_upload):
        # Create an empty DataFrame with the required columns.
        df = pd.DataFrame(columns=["Product Name", "Total Quantity", "Unit Price"])
        filename = "stats"
        
        grouped_stats = generate_numeric_stats(df, filename)
        
        # With an empty DataFrame, no groups are formed.
        self.assertEqual(grouped_stats, {})
        
        # Verify upload_to_gcs was still called with a DataFrame containing an empty "stats" dict.
        mock_upload.assert_called_once()
        uploaded_df = mock_upload.call_args[0][0]
        # Convert the Polars DataFrame to Pandas for inspection.
        uploaded_dict = uploaded_df.to_pandas().iloc[0]["stats"]
        self.assertEqual(uploaded_dict, {})


    def test_empty_dataframe_validate_data(self):
        # Create an empty DataFrame with the required columns.
        df = pd.DataFrame(columns=["Product Name", "Total Quantity", "Date"])
        
        result = validate_data(df)
        
        # The block for empty DataFrame appends an error message to results.
        self.assertIn({"error": "DataFrame is empty, no rows found"}, result["results"])
        self.assertFalse(result["has_errors"])
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["missing_columns"], [])


    @patch("scripts.post_validation.upload_to_gcs", side_effect=Exception("Upload failed"))
    @patch("scripts.post_validation.logger")
    def test_upload_exception(self, mock_logger, mock_upload):
        # Create a valid DataFrame.
        data = {
            "Product Name": ["Apple"],
            "Total Quantity": [10],
            "Unit Price": [1.0],
        }
        df = pd.DataFrame(data)
        filename = "stats"
        
        with self.assertRaises(Exception) as context:
            generate_numeric_stats(df, filename)
        
        self.assertIn("Upload failed", str(context.exception))
        mock_logger.error.assert_called()

    @patch("scripts.post_validation.upload_to_gcs")
    @patch("scripts.post_validation.logger")
    def test_custom_include_columns(self, mock_logger, mock_upload):
        # Create a DataFrame with an extra numeric column.
        data = {
            "Product Name": ["Apple", "Apple", "Banana", "Banana"],
            "Total Quantity": [10, 20, 5, 15],
            "Unit Price": [1.0, 1.2, 0.8, 0.9],
            "Discount": [0.1, 0.2, 0.05, 0.1],
        }
        df = pd.DataFrame(data)
        filename = "stats"
        
        # Specify custom include_columns (only compute stats for "Discount")
        grouped_stats = generate_numeric_stats(df, filename, include_columns=["Discount"])
        
        # For each product, only "Discount" should be computed.
        for product in grouped_stats:
            self.assertIn("Discount", grouped_stats[product])
            self.assertNotIn("Total Quantity", grouped_stats[product])
            self.assertNotIn("Unit Price", grouped_stats[product])


    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.generate_numeric_stats")
    @patch("scripts.post_validation.validate_data")
    def test_main_success(self, mock_validate_data, mock_generate_stats, mock_logger):
        # Simulate a successful validation (no errors)
        mock_validate_data.return_value = {"has_errors": False, "error_count": 0, "missing_columns": []}
        # generate_numeric_stats can return any dictionary
        mock_generate_stats.return_value = {"dummy": "stats"}
        
        # Create a sample Polars DataFrame
        df = pl.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": [10],
        })
        file_name = "output_stats"

        result = main(df, file_name)
        
        # Expect success since there are no errors
        self.assertTrue(result)
        # Verify that the success log was written
        mock_logger.info.assert_any_call("Workflow completed successfully.")

    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.generate_numeric_stats")
    @patch("scripts.post_validation.validate_data")
    def test_main_failure(self, mock_validate_data, mock_generate_stats, mock_logger):
        # Simulate a failed validation (has_errors True)
        mock_validate_data.return_value = {"has_errors": True, "error_count": 2, "missing_columns": ["Total Quantity"]}
        mock_generate_stats.return_value = {"dummy": "stats"}
        
        # Create a sample Pandas DataFrame
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": [10],
        })
        file_name = "output_stats.json"

        result = main(df, file_name)
        
        # Expect failure because errors were detected
        self.assertFalse(result)
        # Even in failure, the workflow is completed successfully (i.e. no exception), so logger.info should have been called.
        mock_logger.info.assert_any_call("Workflow completed successfully.")

    @patch("scripts.post_validation.logger")
    @patch("scripts.post_validation.generate_numeric_stats")
    @patch("scripts.post_validation.validate_data")
    def test_main_exception(self, mock_validate_data, mock_generate_stats, mock_logger):
        # Simulate an exception during validation
        mock_validate_data.side_effect = Exception("Validation crashed")
        file_name = "output_stats.json"
        # Use a sample Pandas DataFrame (or Polars, doesn't matter)
        df = pd.DataFrame({
            "Date": ["2023-01-01"],
            "Product Name": ["Apple"],
            "Total Quantity": [10],
        })

        with self.assertRaises(Exception) as context:
            main(df, file_name)
        
        self.assertIn("Validation crashed", str(context.exception))
        # Verify that logger.error was called with the failure message
        mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
