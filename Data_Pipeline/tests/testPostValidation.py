import unittest
import pandas as pd
from Data_Pipeline.scripts.post_validation import *
from unittest.mock import MagicMock, call, patch

class TestPostiValidation(unittest.TestCase):
    def setUp(self):
        self.error_indices = set()
        self.error_reasons = {}

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


    # @patch("Data_Pipeline.scripts.post_validation.logger")
    # @patch("Data_Pipeline.scripts.post_validation.send_anomaly_alert")
    # @patch("Data_Pipeline.scripts.post_validation.collect_validation_errors")
    # @patch("Data_Pipeline.scripts.post_validation.check_column_types")
    # def test_valid_data_validate_data_post(
    #     self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    # ):
    #     df = pd.DataFrame({
    #         "Date": ["2023-01-01"],
    #         "Product Name": ["Apple"],
    #         "Total Quantity": [10],
    #     })

    #     result = validate_data(df)

    #     self.assertFalse(result["has_errors"])
    #     self.assertEqual(result["error_count"], 0)
    #     self.assertEqual(result["missing_columns"], [])
    #     mock_send_anomaly_alert.assert_not_called()

    
    # @patch("Data_Pipeline.scripts.post_validation.check_column_types")
    # @patch("Data_Pipeline.scripts.post_validation.collect_validation_errors")
    # @patch("Data_Pipeline.scripts.post_validation.send_anomaly_alert")
    # def test_empty_dataframe(
    #     self, mock_send_anomaly_alert, mock_collect_validation_errors, mock_check_column_types
    # ):
    #     df = pd.DataFrame()

    #     result = validate_data(df)

    #     self.assertTrue(result["has_errors"])
    #     self.assertEqual(result["error_count"], 1)
    #     self.assertIn("DataFrame is empty, no rows found", [res["error"] for res in result["results"]])
    #     mock_send_anomaly_alert.assert_called()

    # @patch("Data_Pipeline.scripts.post_validation.logger")
    # @patch("Data_Pipeline.scripts.post_validation.send_anomaly_alert")
    # @patch("Data_Pipeline.scripts.post_validation.collect_validation_errors")
    # @patch("Data_Pipeline.scripts.post_validation.check_column_types")
    # def test_missing_columns(
    #     self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    # ):
    #     df = pd.DataFrame({
    #         "Date": ["2023-01-01"],
    #         "Product Name": ["Apple"],
    #     })  # Missing "Total Quantity"

    #     result = validate_data(df)

    #     self.assertTrue(result["has_errors"])
    #     self.assertGreater(result["error_count"], 0)
    #     self.assertIn("Total Quantity", result["missing_columns"])
    #     mock_collect_validation_errors.assert_called()
    #     mock_send_anomaly_alert.assert_called()


    # @patch("Data_Pipeline.scripts.post_validation.send_anomaly_alert")
    # @patch("Data_Pipeline.scripts.post_validation.collect_validation_errors")
    # @patch("Data_Pipeline.scripts.post_validation.check_column_types")
    # def test_anomalies_detected(
    #     self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert
    # ):
    #     df = pd.DataFrame({
    #         "Date": ["2023-01-01"],
    #         "Product Name": ["Apple"],
    #         "Total Quantity": ["invalid"],  # Should trigger an anomaly
    #     })

    #     def mock_check(df, error_indices, error_reasons):
    #         error_indices.add(0)
    #         error_reasons[0] = ["Total Quantity must be numeric"]

    #     mock_check_column_types.side_effect = mock_check

    #     result = validate_data(df)

    #     self.assertTrue(result["has_errors"])
    #     self.assertEqual(result["error_count"], 1)
    #     mock_send_anomaly_alert.assert_called()

    # @patch("Data_Pipeline.scripts.post_validation.logger")
    # @patch("Data_Pipeline.scripts.post_validation.send_anomaly_alert")
    # @patch("Data_Pipeline.scripts.post_validation.collect_validation_errors")
    # @patch("Data_Pipeline.scripts.post_validation.check_column_types")
    # def test_polars_dataframe(
    #     self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    # ):
    #     df = pl.DataFrame({
    #         "Date": ["2023-01-01"],
    #         "Product Name": ["Apple"],
    #         "Total Quantity": [10],
    #     })

    #     result = validate_data(df)

    #     self.assertFalse(result["has_errors"])
    #     self.assertEqual(result["error_count"], 0)
    #     self.assertEqual(result["missing_columns"], [])
    #     mock_send_anomaly_alert.assert_not_called()

    # @patch("Data_Pipeline.scripts.post_validation.logger")
    # @patch("Data_Pipeline.scripts.post_validation.send_anomaly_alert")
    # @patch("Data_Pipeline.scripts.post_validation.collect_validation_errors")
    # @patch("Data_Pipeline.scripts.post_validation.check_column_types")
    # def test_exception_handling(
    #     self, mock_check_column_types, mock_collect_validation_errors, mock_send_anomaly_alert, mock_logger
    # ):
    #     df = pd.DataFrame({
    #         "Date": ["2023-01-01"],
    #         "Product Name": ["Apple"],
    #         "Total Quantity": [10],
    #     })

    #     mock_check_column_types.side_effect = Exception("Unexpected error")

    #     with self.assertRaises(Exception):
    #         validate_data(df)


if __name__ == "__main__":
    unittest.main()
