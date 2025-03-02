import polars as pl
from polars.testing import assert_frame_equal
from datetime import datetime, timedelta
import io
import unittest
# from scripts.logger import logger
from unittest.mock import MagicMock, patch
from scripts.dataPreprocessing import convert_feature_types, load_bucket_data, detect_anomalies, upload_df_to_gcs, \
    remove_duplicate_records, apply_fuzzy_correction, remove_invalid_records, standardize_product_name, \
    filling_missing_cost_price, convert_string_columns_to_lowercase, compute_most_frequent_price, standardize_date_format, \
    detect_date_order, filling_missing_dates, remove_future_dates, extract_datetime_features, send_anomaly_alert, \
    aggregate_daily_products, extracting_time_series_and_lagged_features, calculate_zscore, iqr_bounds, main

from google.api_core.exceptions import GoogleAPICallError


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.valid_df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        }).with_columns(pl.col("Date").str.to_datetime())

        # Missing features
        self.missing_columns_df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20]
        })


    ### Unit Tests for convert_types.


    def test_convert_feature_types(self):
        # Setup
        df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [10.5, 20.0],
            "Quantity": [1, 2],
            "Transaction ID": ["T001", "T002"],
            "Store Location": ["NY", "LA"],
            "Product Name": ["A", "B"],
            "Producer ID": ["P001", "P002"],
        }).with_columns(pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Test
        converted_df = convert_feature_types(df)

        # Assert
        self.assertEqual(converted_df["Date"].dtype, pl.Datetime)
        self.assertEqual(converted_df["Unit Price"].dtype, pl.Float64)
        self.assertEqual(converted_df["Quantity"].dtype, pl.Int64)
        self.assertEqual(converted_df["Transaction ID"].dtype, pl.Utf8)
        self.assertEqual(converted_df["Store Location"].dtype, pl.Utf8)
        self.assertEqual(converted_df["Product Name"].dtype, pl.Utf8)
        self.assertEqual(converted_df["Producer ID"].dtype, pl.Utf8)


    def test_convert_feature_types_missing_column(self):
        # Setup
        df = pl.DataFrame({"Date": ["2024-01-01"], "Quantity": [5]}).with_columns(
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Test
        converted_df = convert_feature_types(df)

        # Assert
        self.assertIn("Date", converted_df.columns)
        self.assertIn("Quantity", converted_df.columns)
        self.assertEqual(converted_df["Date"].dtype, pl.Datetime)
        self.assertEqual(converted_df["Quantity"].dtype, pl.Int64)

    
    def test_convert_feature_types_empty_dataframe(self):
        # Setup
        df_empty = pl.DataFrame(schema={
            "Date": pl.Datetime,
            "Unit Price": pl.Float64,
            "Quantity": pl.Int64,
            "Transaction ID": pl.Utf8,
            "Store Location": pl.Utf8,
            "Product Name": pl.Utf8,
            "Producer ID": pl.Utf8,
        })

        # Test
        converted_df = convert_feature_types(df_empty)

        # Assert
        self.assertTrue(converted_df.is_empty())
        self.assertEqual(converted_df.schema, df_empty.schema)

    
    def test_convert_feature_types_invalid_data(self):
        # Setup
        df_invalid = pl.DataFrame({"Date": ["invalid_date"], "Quantity": ["not_a_number"]})

        # Test
        with self.assertRaises(Exception):
            convert_feature_types(df_invalid)





    ### Unit tests for standardize_date_format function.

    # Test case where date has multiple formats.
    def test_standardize_date_format_multiple_formats(self):
        # Setup
        data = {
            "Date": [
                "2019-01-03 08:46:08.000",
                "2019-01-03",
                "01-03-2019",
                "03/01/2019",
                "not a date"
            ]
        }
        df = pl.DataFrame(data)

        
        expected_dates = [
            datetime(2019, 1, 3, 8, 46, 8),
            datetime(2019, 1, 3),
            datetime(2019, 1, 3),
            datetime(2019, 1, 3),
            None
        ]

        # Test
        df_result = standardize_date_format(df, date_column="Date")
        
        # Assert
        result_dates = df_result["Date"].to_list()
        self.assertEqual(result_dates, expected_dates, "Date conversion did not produce the expected datetime values.")


    # Test case where date feature is missing.
    def test_standardize_date_format_missing_date_column(self):
        # Setup
        df = pl.DataFrame({
            "Other": ["2019-01-03"]
        })

        # Test
        df_result = standardize_date_format(df, date_column="Date")
        

        # Assert
        self.assertEqual(df_result.to_dicts(), df.to_dicts(), "DataFrame should remain unchanged if date column is missing.")

    
    # Test case where invalid inputs are in date feature.
    def test_standardize_date_format_invalid_inputs(self):
        # Setup
        data = {
            "Date": [
                "08:46:08.000",
                "Invalid date",
                "2019",
                123,
                123.43,
                True,
                "1/1/2000 Date"
            ]
        }
        df = pl.DataFrame(data, strict=False)

        # Expected output dates as Python datetime objects.
        expected_dates = [
            None,
            None,
            None,
            None,
            None,
            None,
            None
        ]


        # Test
        df_result = standardize_date_format(df)

        # Assert
        result_dates = df_result["Date"].to_list()
        self.assertEqual(result_dates, expected_dates, "Date conversion did not produce the expected datetime values.")


    
    # Test case where standardize_date_format handles exception.
    def test_standardize_date_format_exception(self):
        # Setup
        df = pl.DataFrame({"Date": ["2020-01-01", "2020-02-02"]})

        # Test
        with patch.object(df, "with_columns", side_effect=Exception("Forced error")):
            with self.assertRaises(Exception) as context:
                standardize_date_format(df, "Date")

        # Assert
        self.assertEqual(str(context.exception), "Forced error")



    ### Unit tests for detect_date_order function.

    # Test case where date order is Ascending.
    def test_detect_date_order_ascending_order(self):
        # Setup
        df = pl.DataFrame({"Date": ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]})
        df = standardize_date_format(df)

        # Test
        result = detect_date_order(df)

        # Assert
        self.assertEqual(result, "Ascending")


    # Test case where date order is Descending.
    def test_detect_date_order_descending_order(self):
        # Setup
        df = pl.DataFrame({"Date": ["2021-05-01", "2021-04-01", "2021-03-01", "2021-01-01"]})
        df = standardize_date_format(df)

        # Test
        result = detect_date_order(df)

        # Assert
        self.assertEqual(result, "Descending")


    # Test case where date order is Random.
    def test_detect_date_order_random_order(self):
        # Setup
        df = pl.DataFrame({"Date": ["2021-05-01", "2022-04-01", "2020-03-01", "2025-01-01"]})
        df = standardize_date_format(df)

        # Test
        result = detect_date_order(df)

        # Assert
        self.assertEqual(result, "Random")


    # Test empty date column
    def test_empty_date_column(self):
        # Setup
        df = pl.DataFrame({"Date": []})

        # Test
        result = detect_date_order(df, "Date")

        # Assert
        self.assertEqual(result, "Random")


    # Test case where None exist.
    def test_detect_date_order_null_values(self):
        # Setup
        df = pl.DataFrame({"Date": ["2021-01-01", None, None, None, "2021-03-01", "2022-01-01"]})
        df = standardize_date_format(df)
        
        # Test
        result = detect_date_order(df, "Date")

        # Assert
        self.assertEqual(result, "Ascending")

    
    # Test case where there is a single record.
    def test_detect_date_order_single_date(self):
        # Setup
        df = pl.DataFrame({"Date": ["2021-01-01"]})
        df = standardize_date_format(df)

        # Test
        result = detect_date_order(df, "Date")

        # Assert
        self.assertEqual(result, "Random")


    # Test case where function throws exception.
    def test_detect_date_order_exception(self):
        # Setup
        df = pl.DataFrame({"Date": ["2021-01-01", "2021-01-02"]})

        # Test
        with patch.object(pl.DataFrame, "with_columns", side_effect=Exception("Forced error")):
            with self.assertRaises(Exception) as context:
                detect_date_order(df, "Date")

        # Assert
        self.assertEqual(str(context.exception), "Forced error")




    ### Unit Tests for filling_missing_dates function.

    ### Helper function.
    def convert_dates_to_str(self, df: pl.DataFrame, col: str = "Date") -> pl.DataFrame: 
        return df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d").alias(col))
    
    
    # Test case where Ascending order with missing dates, different day records dropped.
    @patch("scripts.dataPreprocessing.detect_date_order", return_value="Ascending")
    @patch("scripts.dataPreprocessing.standardize_date_format",
        side_effect=lambda df, date_column: df.with_columns(
            pl.col(date_column).str.strptime(pl.Datetime, "%Y-%m-%d")
        ))
    def test_filling_missing_dates_ascending(self, mock_standardize, mock_detect):
        # Setup
        data = {
            "Date": ["2020-01-01", None, "2020-01-03", None, "2020-01-05"]
        }
        df = pl.DataFrame(data)

        expected_dates = ["2020-01-01", "2020-01-03", "2020-01-05"]
        
        # Test
        result_df = filling_missing_dates(df, "Date")
        result_df = self.convert_dates_to_str(result_df, "Date")
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_dates, expected_dates)



    # Test case where Descending order with missing dates, different day records dropped.
    @patch("scripts.dataPreprocessing.detect_date_order", return_value="Descending")
    @patch("scripts.dataPreprocessing.standardize_date_format",
        side_effect=lambda df, date_column: df.with_columns(
            pl.col(date_column).str.strptime(pl.Datetime, "%Y-%m-%d")
        ))
    def test_filling_missing_dates_descending(self, mock_standardize, mock_detect):
        # Setup
        data = {
            "Date": ["2020-01-05", None, "2020-01-03", None, "2020-01-01"]
        }
        df = pl.DataFrame(data)

        expected_dates = ["2020-01-05", "2020-01-03", "2020-01-01"]
        
        # Test
        result_df = filling_missing_dates(df, "Date")
        result_df = self.convert_dates_to_str(result_df, "Date")
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_dates, expected_dates)

    
    # Test case where Ascending order with missing dates on same day.
    @patch("scripts.dataPreprocessing.detect_date_order", return_value="Ascending")
    @patch("scripts.dataPreprocessing.standardize_date_format",
       side_effect=lambda df, date_column: df)
    def test_filling_missing_dates_ascending_with_missing_date_same_day(self, mock_standardize, mock_detect):
        # Setup
        date1 = datetime(2022, 1, 1, 10, 0, 0)
        date2 = datetime(2022, 1, 1, 12, 0, 0)
        date_diff_day = datetime(2022, 1, 2, 9, 0, 0)
        
        df = pl.DataFrame({
            "Date": [date1, None, date2]
        })

        expected_date = date1 + (date2 - date1) / 2
        

        # Test
        result_df = filling_missing_dates(df, "Date")
        interpolated = result_df["Date"][1]

        # Assert
        self.assertEqual(result_df.height, 3)
        self.assertAlmostEqual(interpolated.timestamp(), expected_date.timestamp(), delta=1)



    # Test case where Descending order with missing dates on same day.
    @patch("scripts.dataPreprocessing.detect_date_order", return_value="Descending")
    @patch("scripts.dataPreprocessing.standardize_date_format",
       side_effect=lambda df, date_column: df)
    def test_filling_missing_dates_descending_with_missing_date_same_day(self, mock_standardize, mock_detect):
        # Setup
        date1 = datetime(2022, 1, 1, 12, 0, 0)
        date2 = datetime(2022, 1, 1, 10, 0, 0)
        date_diff_day = datetime(2022, 1, 2, 9, 0, 0)
        
        df = pl.DataFrame({
            "Date": [date1, None, date2]
        })

        expected_date = date1 + (date2 - date1) / 2
        

        # Test
        result_df = filling_missing_dates(df, "Date")
        interpolated = result_df["Date"][1]

        # Assert
        self.assertEqual(result_df.height, 3)
        self.assertAlmostEqual(interpolated.timestamp(), expected_date.timestamp(), delta=1)
    


    # Test case where Random order with missing dates.
    @patch("scripts.dataPreprocessing.detect_date_order", return_value="Random")
    @patch("scripts.dataPreprocessing.standardize_date_format",
        side_effect=lambda df, date_column: df.with_columns(
            pl.col(date_column).str.strptime(pl.Datetime, "%Y-%m-%d")
        ))
    def test_filling_missing_dates_random_order(self, mock_standardize, mock_detect):
        # Setup
        data = {
            "Date": ["2022-01-02", None, "2020-01-03", None, "2025-01-01"]
        }
        df = pl.DataFrame(data)

        expected_dates = ["2022-01-02", "2020-01-03", "2025-01-01"]
        
        # Test
        result_df = filling_missing_dates(df, "Date")
        result_df = self.convert_dates_to_str(result_df, "Date")
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_dates, expected_dates)



    # Test case where no missing values in date feature.
    # @patch("scripts.dataPreprocessing.detect_date_order", return_value="Random")
    @patch("scripts.dataPreprocessing.standardize_date_format",
        side_effect=lambda df, date_column: df.with_columns(
            pl.col(date_column).str.strptime(pl.Datetime, "%Y-%m-%d")
        ))
    def test_filling_missing_dates_no_missing_value(self, mock_standardize):
        # Setup
        data = {
            "Date": ["2020-01-05", "2020-01-04", "2020-01-03", "2020-01-02", "2020-01-01"]
        }
        df = pl.DataFrame(data)

        expected_dates = ["2020-01-05", "2020-01-04", "2020-01-03", "2020-01-02", "2020-01-01"]
        
        # Test
        result_df = filling_missing_dates(df, "Date")
        result_df = self.convert_dates_to_str(result_df, "Date")
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_dates, expected_dates)


    
    # Test case where function throws exception.
    @patch("scripts.dataPreprocessing.standardize_date_format",
        side_effect=Exception("Standardization error"))
    def test_filling_missing_dates_throws_exception(self, mock_standardize):
        # Setup
        data = {
            "Date": ["2020-01-05", "2020-01-04", "2020-01-03", "2020-01-02", "2020-01-01"]
        }
        df = pl.DataFrame(data)

        expected_dates = ["2020-01-05", "2020-01-04", "2020-01-03", "2020-01-02", "2020-01-01"]
        
        # Test
        with self.assertRaises(Exception) as context:
            filling_missing_dates(df, "Date")

        # Assert
        self.assertIn("Standardization error", str(context.exception))

    

    ### Unit Tests for remove_future_dates function.

    # Test case where there exist future dates.
    @patch("scripts.dataPreprocessing.datetime")
    def test_remove_future_dates_exists(self, mock_datetime):
        # Setup
        mock_datetime.today.return_value = datetime(2020, 1, 1, 12, 0, 0)
        data = {
            "Date": [
                datetime(2019, 12, 31, 23, 59, 59),  
                datetime(2020, 1, 1, 0, 0, 0),        
                datetime(2020, 1, 1, 12, 0, 0),        
                datetime(2020, 1, 2, 0, 0, 0)
            ]
        }
        df = pl.DataFrame(data)

        expected_dates = [
            datetime(2019, 12, 31, 23, 59, 59),
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 1, 12, 0, 0)
        ]

        # Test
        result_df = remove_future_dates(df)
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_df.height, 3)
        self.assertEqual(expected_dates, result_dates)



    # Test case where no future dates.
    @patch("scripts.dataPreprocessing.datetime")
    def test_remove_future_dates_not_exist(self, mock_datetime):
        # Setup
        mock_datetime.today.return_value = datetime(2020, 1, 1, 12, 0, 0)
        data = {
            "Date": [
                datetime(2019, 12, 31, 23, 59, 59),  
                datetime(2020, 1, 1, 0, 0, 0),        
                datetime(2020, 1, 1, 12, 0, 0),
            ]
        }
        df = pl.DataFrame(data)

        expected_dates = [
            datetime(2019, 12, 31, 23, 59, 59),
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 1, 12, 0, 0)
        ]

        # Test
        result_df = remove_future_dates(df)
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_df.height, 3)
        self.assertEqual(expected_dates, result_dates)


    # Test case where it throws exception.
    @patch("scripts.dataPreprocessing.datetime")
    def test_remove_future_dates_throws_exception(self, mock_datetime):
        # Setup
        mock_datetime.today.side_effect = Exception("Datetime error")
        data = {
            "Date": [
                datetime(2019, 12, 31, 23, 59, 59),  
                datetime(2020, 1, 1, 0, 0, 0),        
                datetime(2020, 1, 1, 12, 0, 0),
            ]
        }
        df = pl.DataFrame(data)

        expected_dates = [
            datetime(2019, 12, 31, 23, 59, 59),
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 1, 12, 0, 0)
        ]

        # Test
        with self.assertRaises(Exception) as context:
            remove_future_dates(df)

        # Assert
        self.assertIn("Datetime error", str(context.exception))



    ### Unit Tests for extract_datetime_features function.

    # Test case where all features are extracted correctly.
    def test_extract_datetime_feature_successfully(self):
        # Setup
        data = {
            "Date": [
                datetime(2020, 1, 1),
                datetime(2020, 6, 15),
                datetime(2020, 12, 31)
            ]
        }
        df = pl.DataFrame(data)

        # Test
        df_features = extract_datetime_features(df)
        years = df_features["Year"].to_list()
        months = df_features["Month"].to_list()
        weeks = df_features["Week_of_year"].to_list()

        expected_years = [2020, 2020, 2020]
        expected_months = [1, 6, 12]
        expected_weeks = [1, 25, 53]

        # Assert
        self.assertEqual(years, expected_years)
        self.assertEqual(months, expected_months)
        self.assertEqual(weeks, expected_weeks)
    


    def test_extract_datetime_features_exception(self):
        # Create a DataFrame with invalid datetime values.
        df = pl.DataFrame({
            "Date": ["not a date", "also not a date"]
        })
        
        # Assert that calling extract_datetime_features raises an Exception.
        with self.assertRaises(Exception):
            extract_datetime_features(df)



    ### Unit tests for compute_most_frequent_price function.


    # Test case where compute_most_frequent_price has [year, month, week].
    def test_compute_most_frequent_price_with_year_month_week(self):
        # Setup
        df = pl.DataFrame({
            "Year":         [2020, 2020, 2020, 2021, 2022, 2022],
            "Month":        [1,    1,    1,    1,    1,    1],
            "Week_of_year": [1,    1,    2,    1,    2,    2],
            "Product Name": ["Milk", "Milk", "Milk", "Milk", "Milk", "Milk"],
            "Unit Price":   [1.5,  1.5,  2.0,  2.0,  2.5,  2.5]
        })
        
        
        # Test
        result = compute_most_frequent_price(df, ["Year", "Month", "Week_of_year"])
        
        expected = pl.DataFrame({
            "Year":         [2020, 2020, 2021, 2022],
            "Month":        [1,    1,    1,    1],
            "Week_of_year": [1,    2,    1,    2],
            "Product Name": ["Milk", "Milk", "Milk", "Milk"],
            "Most_Frequent_Cost": [1.5, 2.0, 2.0, 2.5]
        })
            
        # Sort by Week_of_year for comparison.
        result_sorted = result.sort(["Year", "Month", "Week_of_year", "Product Name"])
        expected_sorted = expected.sort(["Year", "Month", "Week_of_year", "Product Name"])
        assert_frame_equal(result_sorted, expected_sorted, check_dtype=False)


    # Test case where compute_most_frequent_price has [year, month].
    def test_compute_most_frequent_price_with_year_month(self):
        # Setup
        df = pl.DataFrame({
            "Year":         [2020, 2020, 2020, 2020, 2021, 2021, 2022, 2022, 2020],
            "Month":        [1,    1,    2,    2,    1,    2,    3,    3,    1],
            "Product Name": ["Milk", "Milk", "Milk", "Milk", "Bread", "Bread", "Eggs", "Eggs", "Milk"],
            "Unit Price":   [1.5,  2.0,  1.8,  1.8,  1.0,   1.0,   2.5,   2.5,   1.5]
        })

        # Test
        result = compute_most_frequent_price(df, ["Year", "Month"])
        
        expected = pl.DataFrame({
            "Year": [2020, 2020, 2021, 2021, 2022],
            "Month": [1,    2,    1,    2,    3],
            "Product Name": ["Milk", "Milk", "Bread", "Bread", "Eggs"],
            "Most_Frequent_Cost": [1.5, 1.8, 1.0, 1.0, 2.5]
        })

        result_sorted = result.sort(["Year", "Month", "Product Name"])
        expected_sorted = expected.sort(["Year", "Month", "Product Name"])

        # Assert
        assert_frame_equal(result_sorted, expected_sorted, check_dtype=False)



    # Test case where compute_most_frequent_price has [year].
    def test_compute_most_frequent_price_with_year(self):
        # Setup
        df = pl.DataFrame({
            "Year":         [2020, 2020, 2020, 2021, 2021, 2022, 2022, 2020, 2022],
            "Product Name": ["Milk", "Milk", "Milk", "Bread", "Bread", "Eggs", "Eggs", "Milk", "Bread"],
            "Unit Price":   [1.5, 2.0, 1.5, 1.0, 1.0, 2.5, 2.5, 1.5, 1.2]
        })
        
        expected = pl.DataFrame({
            "Year":         [2020, 2021, 2022, 2022],
            "Product Name": ["Milk", "Bread", "Eggs", "Bread"],
            "Most_Frequent_Cost": [1.5, 1.0, 2.5, 1.2]
        })
        
        # Test
        result = compute_most_frequent_price(df, ["Year"])
        result_sorted = result.sort(["Year", "Product Name"])
        expected_sorted = expected.sort(["Year", "Product Name"])
        
        # Assert
        assert_frame_equal(result_sorted, expected_sorted, check_dtype=False)


    # Test case where most_frequent_price is empty.
    def test_compute_most_frequent_price_empty(self):
        # Setup
        df = pl.DataFrame({
            "Year": [2020, 2020],
            "Month": [1, 1],
            "Product Name": ["Milk", "Bread"],
            "Unit Price": [None, None]
        })

        # Test
        result = compute_most_frequent_price(df, ["Year", "Month"])

        # Assert
        self.assertEqual(result.height, 0)


    # Test case where most_frequent_price is tie.
    def test_compute_most_frequent_price_tie(self):
        # Setup
        df = pl.DataFrame({
            "Year": [2020, 2020, 2020, 2020],
            "Month": [1, 1, 1, 1],
            "Product Name": ["Cheese", "Cheese", "Cheese", "Cheese"],
            "Unit Price": [2.0, 3.0, 2.0, 3.0]
        })

        # Test
        result = compute_most_frequent_price(df, ["Year", "Month"])
        mode_val = result["Most_Frequent_Cost"][0]

        # Assert
        self.assertEqual(result.height, 1)
        self.assertIn(mode_val, [2.0, 3.0])

    
    # Test case where compute_most_frequent_price throws an error.
    def test_compute_most_frequent_price_missing_group_column(self):
        # Setup
        df = pl.DataFrame({
            "Year": [2020, 2020],
            "Product Name": ["Milk", "Milk"],
            "Unit Price": [1.5, 1.5]
        })

        # Test
        with self.assertRaises(Exception) as context:
            compute_most_frequent_price(df, ["Year", "Month"])

    

    ### Unit Tests for convert_string_columns_to_lowercase function.

    # Test case where all features are string.
    def test_convert_string_columns_to_lowercase_all_features_string(self):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["Milk", "Beans", "CoFFEE"],
            "Transaction ID": ["XyZ", "12LMN", "OPQ"]
        })

        # Test
        df_result = convert_string_columns_to_lowercase(df)

        # Assert
        self.assertEqual(df_result["Product Name"].to_list(), ["milk", "beans", "coffee"])
        self.assertEqual(df_result["Transaction ID"].to_list(), ["xyz", "12lmn", "opq"])


    # Test case where no features are string.
    def test_convert_string_columns_to_lowercase_no_string(self):
        # Setup
        df = pl.DataFrame({
            "num": [1, 2, 3],
            "flag": [True, False, True]
        })

        # Test
        result_df = convert_string_columns_to_lowercase(df)

        # Assert
        self.assertEqual(df.to_dicts(), result_df.to_dicts())


    # Test case where mix of string and non-string feature.
    def test_convert_string_columns_to_lowercase_mixed_string_features(self):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["MILK", "COFFee"],
            "Unit Price": [30, 25]
        })

        # Test
        result_df = convert_string_columns_to_lowercase(df)

        # Assert
        self.assertEqual(result_df["Product Name"].to_list(), ["milk", "coffee"])
        self.assertEqual(result_df["Unit Price"].to_list(), [30, 25])


    # Test case where empty dataframe.
    def test_convert_string_columns_to_lowercase_empty_dataframe(self):
        # Setup
        df = pl.DataFrame({
            "Product Name": [],
            "Unit Price": []
        })

        # Test
        df_result = convert_string_columns_to_lowercase(df)

        # Assert
        self.assertEqual(df_result.shape, (0, 2))


    # Test case where exception raised
    @patch("polars.Expr.str")
    def test_convert_string_column_to_lowercase_exception(self, mock_str):
        # Setup
        mock_str.to_lowercase.side_effect = Exception("Test error")
        df = pl.DataFrame({
            "Product Name": ["MILK", "COFFee"]
        })

        # Test
        with self.assertRaises(Exception) as context:
            convert_string_columns_to_lowercase(df)

        # Assert
        self.assertIn("Test error", str(context.exception))



    
    ### Unit tests for filling_missing_cost_price functions.

    
    # Test case where cost price filled by Week.
    @patch("scripts.dataPreprocessing.extract_datetime_features")
    @patch("scripts.dataPreprocessing.compute_most_frequent_price")
    def test_cost_price_filled_by_week(self, mock_compute_most_frequent_price, mock_extract_datetime_features):
        # Setup
        df = pl.DataFrame({
            "Year": [2023, 2023],
            "Month": [1, 1],
            "Week_of_year": [2, 3],
            "Product Name": ["milk", "bread"],
            "Unit Price": [None, 20.0]
        })

        # Patch extract_datetime_features to return the same DataFrame
        mock_extract_datetime_features.return_value = df

        # Created a valid week-level aggregation for "milk"
        price_by_week_df = pl.DataFrame({
            "Year": [2023],
            "Month": [1],
            "Week_of_year": [2],
            "Product Name": ["milk"],
            "Most_Frequent_Cost": [15.0]
        })
        
        price_by_month_df = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })

        price_by_year_df = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })


        mock_compute_most_frequent_price.side_effect = [
            price_by_week_df, price_by_month_df, price_by_year_df
        ]

        # Test
        df_result = filling_missing_cost_price(df)

        # Assert
        self.assertEqual(df_result["Unit Price"].to_list(), [15.0, 20.0])



    # Test case where cost price filled by Week, Month.
    @patch("scripts.dataPreprocessing.extract_datetime_features")
    @patch("scripts.dataPreprocessing.compute_most_frequent_price")
    def test_cost_price_filled_by_week_month(self, mock_compute_most_frequent_price, mock_extract_datetime_features):
        # Setup
        df = pl.DataFrame({
            "Year": [2023, 2023],
            "Month": [1, 1],
            "Week_of_year": [2, 3],
            "Product Name": ["milk", "bread"],
            "Unit Price": [None, None]
        })
        

        mock_extract_datetime_features.return_value = df

        # Week-level aggregation returns a value for "milk"
        price_by_week = pl.DataFrame({
            "Year": [2023],
            "Month": [1],
            "Week_of_year": [2],
            "Product Name": ["milk"],
            "Most_Frequent_Cost": [15.0]
        })
        # Month-level aggregation returns a value for "bread"
        price_by_month = pl.DataFrame({
            "Year": [2023],
            "Month": [1],
            "Product Name": ["bread"],
            "Most_Frequent_Cost": [18.0]
        })
        # Year-level aggregation: return an empty DataFrame with proper schema.
        price_by_year = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })
        mock_compute_most_frequent_price.side_effect = [price_by_week, price_by_month, price_by_year]


        # Test
        df_result = filling_missing_cost_price(df)

        # Assert
        self.assertEqual(df_result["Unit Price"].to_list(), [15.0, 18.0])


    # Test case where cost price filled by Week, Month, Year.
    @patch("scripts.dataPreprocessing.extract_datetime_features")
    @patch("scripts.dataPreprocessing.compute_most_frequent_price")
    def test_cost_price_filled_by_week_month_year(self, mock_compute_most_frequent_price, mock_extract_datetime_features):
        # Setup
        df = pl.DataFrame({
            "Year": [2023],
            "Month": [1],
            "Week_of_year": [2],
            "Product Name": ["milk"],
            "Unit Price": [None]
        })
        mock_extract_datetime_features.return_value = df

        # Week-level: empty DataFrame with correct schema.
        price_by_week = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Week_of_year": pl.Series("Week_of_year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })
        # Month-level: empty.
        price_by_month = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })
        # Year-level: returns a value.
        price_by_year = pl.DataFrame({
            "Year": [2023],
            "Product Name": ["milk"],
            "Most_Frequent_Cost": [20.0]
        })
        mock_compute_most_frequent_price.side_effect = [price_by_week, price_by_month, price_by_year]

        # Test
        df_result = filling_missing_cost_price(df)

        # Assert
        self.assertEqual(df_result["Unit Price"].to_list(), [20.0])



    # Test case where no missing cost price.
    @patch("scripts.dataPreprocessing.extract_datetime_features")
    @patch("scripts.dataPreprocessing.compute_most_frequent_price")
    def test_cost_price_filled_no_missing_cost_price(self, mock_compute_most_frequent_price, mock_extract_datetime_features):
        # Setup
        df = pl.DataFrame({
            "Year": [2023, 2023],
            "Month": [1, 1],
            "Week_of_year": [2, 3],
            "Product Name": ["milk", "bread"],
            "Unit Price": [10.0, 20.0]
        })
        mock_extract_datetime_features.return_value = df
        
        empty_week = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Week_of_year": pl.Series("Week_of_year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })

        empty_month = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })

        empty_year = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })
        mock_compute_most_frequent_price.side_effect = [empty_week, empty_month, empty_year]


        # Test
        df_result = filling_missing_cost_price(df)

        # Assert
        self.assertEqual(df_result["Unit Price"].to_list(), [10.0, 20.0])




    # Test case where cost price filled by default value.
    @patch("scripts.dataPreprocessing.extract_datetime_features")
    @patch("scripts.dataPreprocessing.compute_most_frequent_price")
    def test_cost_price_filled_by_default_value(self, mock_compute_most_frequent_price, mock_extract_datetime_features):
        # Setup
        df = pl.DataFrame({
            "Year": [2023, 2023],
            "Month": [1, 1],
            "Week_of_year": [2, 3],
            "Product Name": ["milk", "bread"],
            "Unit Price": [None, None]
        })
        mock_extract_datetime_features.return_value = df

        empty_week = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Week_of_year": pl.Series("Week_of_year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })

        empty_month = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Month": pl.Series("Month", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })

        empty_year = pl.DataFrame({
            "Year": pl.Series("Year", [], dtype=pl.Int64),
            "Product Name": pl.Series("Product Name", [], dtype=pl.Utf8),
            "Most_Frequent_Cost": pl.Series("Most_Frequent_Cost", [], dtype=pl.Float64)
        })

        mock_compute_most_frequent_price.side_effect = [empty_week, empty_month, empty_year]

        
        # Test
        df_result = filling_missing_cost_price(df)

        # Assert
        self.assertEqual(df_result["Unit Price"].to_list(), [0, 0])



    
    # Test case where exception handle.
    @patch("scripts.dataPreprocessing.extract_datetime_features", side_effect=Exception("Test Exception"))
    def test_cost_price_filled_throws_exception(self, mock_extract_datetime_features):
        # Setup
        df = pl.DataFrame({
            "Year": [2023],
            "Month": [1],
            "Week_of_year": [2],
            "Product Name": ["milk"],
            "Unit Price": [None]
        })


        # Test
        with self.assertRaises(Exception) as context:
            filling_missing_cost_price(df)
        

        # Assert
        self.assertIn("Test Exception", str(context.exception))



    ### Unit tests remove_invalid_records functions.

    # Test case where records are valid.
    def test_remove_invalid_records_valid_records(self):
        # Setup
        df = pl.DataFrame({
            "Quantity": [10, 20, 30],
            "Product Name": ["milk", "bread", "cheese"],
            "Unit Price": [1, 2, 3]
        })

        # Test
        result_df = remove_invalid_records(df)

        # Assert
        self.assertEqual(result_df.to_dicts(), df.to_dicts())

    # Test case where missing Quantity.
    def test_remove_invalid_records_missing_Quantity(self):
        # Setup
        df = pl.DataFrame({
            "Quantity": [10, None, 30],
            "Product Name": ["milk", "bread", "cheese"]
        })

        expected_df = pl.DataFrame({
            "Quantity": [10, 30],
            "Product Name": ["milk", "cheese"]
        })

        # Test
        result_df = remove_invalid_records(df)

        # Assert
        self.assertEqual(result_df.to_dicts(), expected_df.to_dicts())



    # Test case where missing Product Name.
    def test_remove_invalid_records_missing_Product_Name(self):
        # Setup
        df = pl.DataFrame({
            "Quantity": [10, 20, 30],
            "Product Name": ["milk", None, "cheese"]
        })
        expected_df = pl.DataFrame({
            "Quantity": [10, 30],
            "Product Name": ["milk", "cheese"]
        })

        # Test
        result_df = remove_invalid_records(df)

        # Assert
        self.assertEqual(expected_df.to_dicts(), result_df.to_dicts())


    # Test case where missing both Quantity and Product Name.
    def test_remove_invalid_records_missing_both_Quantity_Product_Name(self):
        # Setup
        df = pl.DataFrame({
            "Quantity": [10, None, 30, None],
            "Product Name": ["milk", "bread", None, None]
        })

        expected_df = pl.DataFrame({
            "Quantity": [10],
            "Product Name": ["milk"]
        })

        # Test
        result_df = remove_invalid_records(df)

        # Assert
        self.assertEqual(result_df.to_dicts(), expected_df.to_dicts())


    # Test case where empty dataframe.
    def test_remove_invalid_records_empty_dataframe(self):
        # Setup
        df = pl.DataFrame({
            "Quantity": [],
            "Product Name": []
        })


        # Test
        result_df = remove_invalid_records(df)

        # Assert
        self.assertEqual(result_df.shape, (0,2))


    # Test case where it handles any exception.
    def test_remove_invalid_records_handles_exception(self):
        # Setup
        df = pl.DataFrame({
            "Quantity": [10, 20, 30]
        }) 

        # Test
        with self.assertRaises(Exception) as context:
            remove_invalid_records(df)

        # Assert
        self.assertIn("Product Name", str(context.exception))


    
    ### Unit tests for standardize_product_name function.

    # Test case where standardize_product_name executes successfully.
    def test_standardize_product_name_executes_successfully(self):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["  MILK  ", "C0FFEE", "Bread@Home", "T0M@TO"]
        })

        expected_df = ["milk", "coffee", "breadahome", "tomato"]

        # Test
        result_df = standardize_product_name(df)

        # Assert
        self.assertEqual(result_df["Product Name"].to_list(), expected_df)


    # Test case where empty string.
    def test_standardize_product_name_with_empty_string(self):
        # Setup
        df = pl.DataFrame({
            "Product Name": [""]
        })

        # Test
        result_df = standardize_product_name(df)

        # Assert
        self.assertEqual(result_df["Product Name"].to_list(), [""])


    # Test case where None value.
    def test_standardize_product_name_with_None_value(self):
        # Setup
        df = pl.DataFrame({
            "Product Name": [None]
        })

        # Test
        result_df = standardize_product_name(df)

        # Assert
        self.assertTrue(result_df["Product Name"][0] is None)


    # Test case where exception handling is done.
    @patch("scripts.dataPreprocessing.pl.col", side_effect=Exception("Test error"))
    def test_standardize_product_name_with_exception_handling(self, mock_to_lowercase):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["MILK"]
        })

        # Test
        with self.assertRaises(Exception) as context:
            standardize_product_name(df)

        # Assert
        self.assertIn("Test error", str(context.exception))




    ### Unit tests for apply_fuzzy_correction function.

    # Test case where match is found.
    @patch("scripts.dataPreprocessing.process.extractOne")
    def test_apply_fuzzy_correction_match_found(self, mock_extractOne):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["MILK", "BRED", "coffee"]  # "coffee" should remain unchanged.
        })
        reference_list = ["milk", "bread", "coffee"]

        mock_extractOne.side_effect = [
            ("milk", 95, 0),
            ("bread", 85, 1), 
            ("coffee", 100, 2)
        ]

        expected_df = ["milk", "bread", "coffee"]

        # Test
        result_df = apply_fuzzy_correction(df, reference_list, threshold=80)

        # Assert
        self.assertEqual(result_df["Product Name"].to_list(), expected_df)

    
    # Test case where below threshold with no correction
    @patch("scripts.dataPreprocessing.process.extractOne")
    def test_apply_fuzzy_correction_below_threshold_with_no_correction(self, mock_extractOne):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["mlk"]
        })
        reference_list = ["milk"]

        mock_extractOne.return_value = ("milk", 95, 0)

        # Test
        result_df = apply_fuzzy_correction(df, reference_list, threshold=100)

        # Assert
        self.assertEqual(result_df["Product Name"].to_list(), ["mlk"])


    # Test case where empty dataframe.
    @patch("scripts.dataPreprocessing.process.extractOne")
    def test_apply_fuzzy_correction_no_dataframe(self, mock_extractOne):
        # Setup
        df = pl.DataFrame({"Product Name": []})
        reference_list = ["milk", "bread"]

        # Test
        result_df = apply_fuzzy_correction(df, reference_list)

        # Assert
        self.assertEqual(result_df.height, 0)


    # Test case where exception is handled.
    @patch("scripts.dataPreprocessing.process.extractOne", side_effect=Exception("Test error"))
    def test_apply_fuzzy_correction_exception_handling(self, mock_extractOne):
        # Setup
        df = pl.DataFrame({
            "Product Name": ["MILK"]
        })
        reference_list = ["milk"]

        # Test
        with self.assertRaises(Exception) as context:
            apply_fuzzy_correction(df, reference_list)

        # Assert
        self.assertIn("Test error", str(context.exception))
    


    ### Unit tests for remove_duplicate_records function.

    # Test case where duplicte records exist.
    def test_remove_duplicate_records_duplicate_records_exist(self):
        # Setup
        df = pl.DataFrame({
            "Transaction ID": ["T001", "T002", "T001", "T003", "T002"],
            "Value": [100, 200, 150, 300, 250]
        })

        expected_df = pl.DataFrame({
            "Transaction ID": ["T001", "T002", "T003"],
            "Value": [100, 200, 300]
        })

        # Test
        result_df = remove_duplicate_records(df)
        
        # Assert
        self.assertEqual(result_df.to_dicts(), expected_df.to_dicts())


    # Test case where no duplicate records exist.
    def test_remove_duplicate_records_no_duplicate_records(self):
        # Setup
        df = pl.DataFrame({
            "Transaction ID": ["T001", "T002", "T003"],
            "Value": [100, 200, 300]
        })


        # Test
        result_df = remove_duplicate_records(df)

        # Assert
        self.assertEqual(df.to_dicts(), result_df.to_dicts())


    # Test case where empty dataframe.
    def test_remove_duplicate_records_empty_dataframe(self):
        # Setup
        df = pl.DataFrame({
            "Transaction ID": [],
            "Value": []
        })

        # Test
        result_df = remove_duplicate_records(df)

        # Assert
        self.assertEqual(df.to_dicts(), result_df.to_dicts())

    
    # Test case where exception is handled.
    @patch("polars.DataFrame.unique", side_effect=Exception("Test error"))
    def test_remove_duplicate_records_exception_handling(self, mock_unique):
        # Setup
        df = pl.DataFrame({
            "Transaction ID": ["T001", "T002", "T001"],
            "Value": [100, 200, 150]
        })


        # Test
        with self.assertRaises(Exception) as context:
            remove_duplicate_records(df)
        
        # Assert
        self.assertIn("Test error", str(context.exception))



    ### Unit tests for send_anomaly_alert function.

    # Test case where no anomalies.
    @patch("scripts.dataPreprocessing.send_email", side_effect=True)
    @patch('scripts.dataPreprocessing.logger')
    def test_send_anomaly_alert_no_anomalies(self, mock_logger, mock_send_email):
        # Setup
        anomalies = {
            "price_anomalies": pl.DataFrame(), 
            "quantity_anomalies": pl.DataFrame()
        }

        # Test 
        send_anomaly_alert(anomalies, recipient="test@example.com", subject="Alert")

        # Assert
        mock_send_email.assert_not_called()
        mock_logger.info.assert_called_with("No anomalies detected; no alert email sent.")


    # Test case where with anomalies.
    @patch("scripts.dataPreprocessing.send_email")
    @patch("scripts.dataPreprocessing.logger")
    def test_send_anomaly_alert_with_anomalies(self, mock_logger, mock_send_email):
        # Setup
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {
            "price_anomalies": pl.DataFrame(), 
            "quantity_anomalies": non_empty_df
        }

        mock_send_email.return_value = None

        # Test
        send_anomaly_alert(anomalies, recipient="test@example.com", subject="Alert")

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
        self.assertTrue("quantity_anomalies" in attachment_df["anomaly_type"].values)
        mock_logger.info.assert_any_call("Anomaly alert email sent.")
    

    # Test case where Exception handled.
    @patch("scripts.dataPreprocessing.send_email", side_effect=Exception("Error sending an alert email."))
    @patch('scripts.dataPreprocessing.logger')
    def test_send_anomaly_alert_throws_exception(self, mock_logger, mock_send_email):
        # Setup
        data = {"col": [1, 2, 3]}
        non_empty_df = pl.DataFrame(data)
        anomalies = {
            "price_anomalies": non_empty_df
        }

        # Test
        send_anomaly_alert(anomalies, recipient="test@example.com", subject="Alert")

        # Assert
        mock_send_email.assert_called_once()
        mock_logger.error.assert_called_with("Error sending an alert email: Error sending an alert email.")

    

    ### Unit tests for aggregate_daily_products function.

    # Test case where aggregation is done.
    def test_aggregate_daily_products_multiple_records(self):
        # Setup
        df = pl.DataFrame({
            "Date": [
                "2023-01-01", "2023-01-01", "2023-01-01",
                "2023-01-02", "2023-01-02",
                "2023-01-01"
            ],
            "Product Name": [
                "milk", "milk", "coffee",
                "milk", "coffee",
                "milk"
            ],
            "Unit Price": [
                2.5, 2.5, 3.0,
                2.5, 3.0,
                2.5
            ],
            "Quantity": [
                10, 5, 8,
                7, 6,
                3
            ]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d"))  

        expected_df = pl.DataFrame({
            "Date": [
                "2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"
            ],
            "Product Name": [
                "coffee", "milk", "coffee", "milk"
            ],
            "Unit Price": [
                3.0, 2.5, 3.0, 2.5
            ],
            "Total Quantity": [
                8, 18, 6, 7
            ]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d").dt.date()) 

        # Test
        result_df = aggregate_daily_products(df).sort(["Date", "Product Name"])

        # Assert
        assert_frame_equal(result_df, expected_df)



    # Test case where single record.
    def test_aggregate_daily_products_single_record(self):
        # Setup
        df = pl.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "Product Name": ["milk", "coffee"],
            "Unit Price": [2.5, 3.0],
            "Quantity": [10, 8]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d"))  

        expected_df = pl.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "Product Name": ["milk", "coffee"],
            "Unit Price": [2.5, 3.0],
            "Total Quantity": [10, 8]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d").dt.date())  

        # Test
        result_df = aggregate_daily_products(df).sort(["Date", "Product Name"])

        # Assert
        assert_frame_equal(result_df, expected_df)


    # Test case where empty dataframe.
    def test_aggregate_daily_products_empty_dataframe(self):
        # Setup
        df = pl.DataFrame({
            "Date": [], "Product Name": [], "Unit Price": [], "Quantity": []
        }).with_columns(pl.col("Date").cast(pl.Date))  

        expected_df = pl.DataFrame({
            "Date": [], "Product Name": [], "Unit Price": [], "Total Quantity": []
        }).with_columns(pl.col("Date").cast(pl.Date)) 

        # Test
        result_df = aggregate_daily_products(df)

        # Assert
        self.assertEqual(result_df.height, 0)
        self.assertListEqual(result_df.columns, ["Date", "Product Name", "Unit Price", "Total Quantity"])



class TestTimeSeriesFeatureExtraction(unittest.TestCase):
    # Setup
    def setUp(self):
        """Set up a sample dataframe for testing."""
        base_date = datetime(2024, 1, 1)
        self.df = pl.DataFrame({
            "Date": [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)],
            "Product Name": ["A"] * 10,
            "Total Quantity": list(range(10))
        })
        self.df = self.df.with_columns(pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d"))

    

    # Test case where datetime_extraction_and_lagged_features executes successfully.
    def test_datetime_extraction_executes_successfully(self):
        # Test
        result_df = extracting_time_series_and_lagged_features(self.df)
        
        # Assert
        self.assertIn("day_of_week", result_df.columns)
        self.assertIn("is_weekend", result_df.columns)
        self.assertIn("lag_1", result_df.columns)
        self.assertIn("lag_7", result_df.columns)
        self.assertIn("rolling_mean_7", result_df.columns)

    
    # Test case where datetime_extraction_and_lagged_features
    def test_lagged_features_executes_successfully(self):
        # Test
        result_df = extracting_time_series_and_lagged_features(self.df)
        expected_lag_1 = [None] + list(range(9))  # First row should be None
        expected_lag_7 = [None] * 7 + list(range(3))  # First 7 rows should be None

        # Assert
        
        self.assertEqual(result_df["lag_1"].to_list(), expected_lag_1)
        self.assertEqual(result_df["lag_7"].to_list(), expected_lag_7)

    
    def test_rolling_average(self):
        # Test
        result_df = extracting_time_series_and_lagged_features(self.df)
        expected_rolling = [None] * 6 + [sum(range(i-6, i+1)) / 7 for i in range(6, 10)]

        # Assert
        self.assertEqual(result_df["rolling_mean_7"].to_list(), expected_rolling)


    def test_multiple_products(self):
        # Setup
        df_multi = pl.concat([
            self.df,
            self.df.with_columns(pl.lit("B").alias("Product Name"))
        ])

        # Test
        result_df = extracting_time_series_and_lagged_features(df_multi)

        # Assert
        self.assertEqual(result_df.filter(pl.col("Product Name") == "A")["lag_1"].to_list(), result_df.filter(pl.col("Product Name") == "B")["lag_1"].to_list())

    # Test case where dataframe is empty.
    def test_empty_dataframe(self):
        # Setup
        df_empty = pl.DataFrame({"Date": [], "Product Name": [], "Total Quantity": []})

        # Test
        result_df = extracting_time_series_and_lagged_features(df_empty)

        # Assert
        self.assertTrue(result_df.is_empty())


    def test_missing_values(self):
        # Setup
        df_missing = self.df.with_columns(pl.lit(None).cast(pl.Int64).alias("Total Quantity"))

        # Test
        result_df = extracting_time_series_and_lagged_features(df_missing)

        # Assert
        self.assertTrue(result_df["lag_1"].null_count() > 0)
        self.assertTrue(result_df["rolling_mean_7"].null_count() > 0)


    def test_datetime_extraction_exception(self):
        """Test exception handling during datetime feature extraction."""
        df_invalid = self.df.with_columns(pl.lit(None).alias("Date"))
        with self.assertRaises(Exception):
            extracting_time_series_and_lagged_features(df_invalid)

    def test_lagged_feature_exception(self):
        """Test exception handling during lagged feature computation."""
        df_invalid = self.df.drop("Total Quantity")
        with self.assertRaises(Exception):
            extracting_time_series_and_lagged_features(df_invalid)

    

    ### Unit Tests for z-score function.

    def test_calculate_zscore(self):
        # Setup
        series = pl.Series("values", [10, 20, 30, 40, 50])

        # Test
        zscores = calculate_zscore(series)

        # Assert
        self.assertAlmostEqual(zscores.mean(), 0, places=6)
        self.assertAlmostEqual(zscores.std(), 1, places=6)

    # Test case where z-score is contant.
    def test_calculate_zscore_constant_series(self):
        # Setup
        series = pl.Series("values", [5, 5, 5, 5, 5])

        # Test
        zscores = calculate_zscore(series)

        # Assert
        self.assertTrue(all(z == 0 for z in zscores))

    # Test case where series has single value.
    def test_calculate_zscore_single_value(self):
        # Setup
        series = pl.Series("values", [100])

        # Test
        zscores = calculate_zscore(series)

        # Assert
        self.assertEqual(zscores[0], 0)


    # Test case where z-score with series has None value.
    def test_calculate_zscore_with_nan(self):
        # Setup
        series = pl.Series("values", [10, 20, None, 30, 40])

        # Test
        zscores = calculate_zscore(series.drop_nulls())

        # Assert
        self.assertAlmostEqual(zscores.mean(), 0, places=6)
        self.assertAlmostEqual(zscores.std(), 1, places=6)

    # Test case where z-score throws exception.
    def test_calculate_zscore_throws_exception(self):
        # Test
        with self.assertRaises(Exception):
            calculate_zscore(pl.Series("values", ["a", "b", "c"]))

    
    def test_iqr_bounds_data(self):
        # Setup
        series = pl.Series([10, 12, 15, 14, 18, 20, 25, 30])

        # Test
        lower, upper = iqr_bounds(series)

        # Assert
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
        self.assertGreaterEqual(lower, 0)

    
    def test_iqr_bounds_identical_values(self):
        # Setup
        series = pl.Series([5, 5, 5, 5, 5])

        # Test
        lower, upper = iqr_bounds(series)

        # Assert
        self.assertEqual(lower, 5) 
        self.assertEqual(upper, 5)


    def test_iqr_bounds_one_value(self):
        # Setup
        series = pl.Series([10])

        # Test
        lower, upper = iqr_bounds(series)

        # Assert
        self.assertEqual(lower, 10) 
        self.assertEqual(upper, 10)

    
    def test_iqr_bounds_negative_values(self):
        # Setup
        series = pl.Series([-10, -5, 0, 5, 10])

        # Test
        lower, upper = iqr_bounds(series)

        # Assert
        self.assertLess(lower, 0)
        self.assertGreater(upper, 10)

    
    def test_iqr_bounds_zero_and_mixed_values(self):
        # Setup
        series = pl.Series([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])

        # Test
        lower, upper = iqr_bounds(series)

        # Assert
        self.assertLess(lower, 0) 
        self.assertGreater(upper, 5)

    
    def test_iqr_bounds_empty_series(self):
        # Setup
        series = pl.Series([])

        # Test
        with self.assertRaises(ValueError):
            iqr_bounds(series)



    ### Unit Tests for main function.

    # Test case where main function executes successfully.
    @patch("scripts.dataPreprocessing.upload_df_to_gcs")
    @patch("scripts.dataPreprocessing.save_cleaned_data")
    @patch("scripts.dataPreprocessing.extracting_time_series_and_lagged_features")
    @patch("scripts.dataPreprocessing.aggregate_daily_products")
    @patch("scripts.dataPreprocessing.send_anomaly_alert")
    @patch("scripts.dataPreprocessing.detect_anomalies", return_value=({}, pl.DataFrame()))
    @patch("scripts.dataPreprocessing.remove_duplicate_records")
    @patch("scripts.dataPreprocessing.remove_invalid_records")
    @patch("scripts.dataPreprocessing.filling_missing_cost_price")
    @patch("scripts.dataPreprocessing.apply_fuzzy_correction")
    @patch("scripts.dataPreprocessing.standardize_product_name")
    @patch("scripts.dataPreprocessing.convert_string_columns_to_lowercase")
    @patch("scripts.dataPreprocessing.convert_feature_types")
    @patch("scripts.dataPreprocessing.filling_missing_dates")
    @patch("scripts.dataPreprocessing.load_bucket_data")
    @patch("scripts.dataPreprocessing.load_data")
    def test_main_successful_execution(
        self, mock_load_data, mock_load_bucket_data, mock_filling_missing_dates,
        mock_convert_feature_types, mock_convert_string_columns_to_lowercase, mock_standardize_product_name,
        mock_apply_fuzzy_correction, mock_filling_missing_cost_price, mock_remove_invalid_records,
        mock_remove_duplicate_records, mock_detect_anomalies, mock_send_anomaly_alert,
        mock_aggregate_daily_products, mock_extracting_features, mock_save_cleaned_data,
        mock_upload_df_to_gcs
    ):
        # Setup

        # Mock dataframe
        mock_df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d"))

        mock_load_data.return_value = mock_df
        mock_load_bucket_data.return_value = mock_df

        # Mock all transformations to return the modified dataframe
        for mock_func in [
            mock_filling_missing_dates, mock_convert_feature_types, mock_convert_string_columns_to_lowercase,
            mock_standardize_product_name, mock_apply_fuzzy_correction, mock_filling_missing_cost_price,
            mock_remove_invalid_records, mock_remove_duplicate_records, mock_aggregate_daily_products,
            mock_extracting_features
        ]:
            mock_func.return_value = mock_df

        # Test
        main(input_file="input.xlsx", output_file="output.csv", cloud=False)

        # Assert
        mock_load_data.assert_called_once()
        mock_filling_missing_dates.assert_called_once()
        mock_convert_feature_types.assert_called_once()
        mock_convert_string_columns_to_lowercase.assert_called_once()
        mock_standardize_product_name.assert_called_once()
        mock_apply_fuzzy_correction.assert_called_once()
        mock_filling_missing_cost_price.assert_called_once()
        mock_remove_invalid_records.assert_called_once()
        mock_remove_duplicate_records.assert_called_once()
        mock_detect_anomalies.assert_called_once()
        mock_send_anomaly_alert.assert_called_once()
        mock_aggregate_daily_products.assert_called_once()
        mock_extracting_features.assert_called_once()
        mock_save_cleaned_data.assert_called_once()

    

    @patch("scripts.dataPreprocessing.load_data", side_effect=Exception("File not found"))
    def test_main_load_data_failure(self, mock_load_data):
        # Test
        with self.assertRaises(Exception) as context:
            main(input_file="invalid.xlsx", output_file="output.csv", cloud=False)

        # Assert
        self.assertIn("File not found", str(context.exception))
        mock_load_data.assert_called_once()


    @patch("scripts.dataPreprocessing.upload_df_to_gcs")
    @patch("scripts.dataPreprocessing.save_cleaned_data")
    @patch("scripts.dataPreprocessing.extracting_time_series_and_lagged_features")
    @patch("scripts.dataPreprocessing.aggregate_daily_products")
    @patch("scripts.dataPreprocessing.send_anomaly_alert")
    @patch("scripts.dataPreprocessing.remove_duplicate_records")
    @patch("scripts.dataPreprocessing.remove_invalid_records")
    @patch("scripts.dataPreprocessing.filling_missing_cost_price")
    @patch("scripts.dataPreprocessing.apply_fuzzy_correction")
    @patch("scripts.dataPreprocessing.standardize_product_name")
    @patch("scripts.dataPreprocessing.convert_string_columns_to_lowercase")
    @patch("scripts.dataPreprocessing.convert_feature_types")
    @patch("scripts.dataPreprocessing.filling_missing_dates")
    @patch("scripts.dataPreprocessing.load_bucket_data")
    @patch("scripts.dataPreprocessing.load_data", return_value=pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d")))
    @patch("scripts.dataPreprocessing.detect_anomalies", side_effect=Exception("Anomaly detection failed"))
    def test_main_anomaly_detection_failure(
        self, mock_detect_anomalies, mock_load_data, mock_load_bucket_data, 
        mock_filling_missing_dates, mock_convert_feature_types, 
        mock_convert_string_columns_to_lowercase, mock_standardize_product_name, 
        mock_apply_fuzzy_correction, mock_filling_missing_cost_price, 
        mock_remove_invalid_records, mock_remove_duplicate_records, 
        mock_aggregate_daily_products, mock_extracting_time_series_and_lagged_features, 
        mock_send_anomaly_alert, mock_save_cleaned_data, mock_upload_df_to_gcs
    ):
        # Test

        with self.assertRaises(Exception) as context:
            main(input_file="input.xlsx", output_file="output.csv", cloud=False)

        # Assert
        self.assertIn("Anomaly detection failed", str(context.exception))        
        mock_load_data.assert_called_once()
        mock_detect_anomalies.assert_called_once()
        mock_filling_missing_dates.assert_called_once()
        mock_convert_feature_types.assert_called_once()
        mock_convert_string_columns_to_lowercase.assert_called_once()
        mock_standardize_product_name.assert_called_once()
        mock_apply_fuzzy_correction.assert_called_once()
        mock_filling_missing_cost_price.assert_called_once()
        mock_remove_invalid_records.assert_called_once()
        mock_remove_duplicate_records.assert_called_once()
        mock_aggregate_daily_products.assert_not_called()
        mock_extracting_time_series_and_lagged_features.assert_not_called()
        mock_send_anomaly_alert.assert_not_called()
        mock_save_cleaned_data.assert_not_called()
        mock_upload_df_to_gcs.assert_not_called()
        mock_load_bucket_data.assert_not_called()

        

    @patch("scripts.dataPreprocessing.upload_df_to_gcs")
    @patch("scripts.dataPreprocessing.save_cleaned_data")
    @patch("scripts.dataPreprocessing.extracting_time_series_and_lagged_features")
    @patch("scripts.dataPreprocessing.aggregate_daily_products")
    @patch("scripts.dataPreprocessing.send_anomaly_alert")
    @patch("scripts.dataPreprocessing.detect_anomalies", return_value=({}, pl.DataFrame()))
    @patch("scripts.dataPreprocessing.remove_duplicate_records")
    @patch("scripts.dataPreprocessing.remove_invalid_records")
    @patch("scripts.dataPreprocessing.filling_missing_cost_price")
    @patch("scripts.dataPreprocessing.apply_fuzzy_correction")
    @patch("scripts.dataPreprocessing.standardize_product_name")
    @patch("scripts.dataPreprocessing.convert_string_columns_to_lowercase")
    @patch("scripts.dataPreprocessing.convert_feature_types")
    @patch("scripts.dataPreprocessing.filling_missing_dates")
    @patch("scripts.dataPreprocessing.load_bucket_data", return_value=pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Unit Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        }).with_columns(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d")))
    def test_main_cloud_execution(
        self, mock_load_bucket_data, mock_filling_missing_dates, mock_convert_feature_types,
        mock_convert_string_columns_to_lowercase, mock_standardize_product_name,
        mock_apply_fuzzy_correction, mock_filling_missing_cost_price, mock_remove_invalid_records,
        mock_remove_duplicate_records, mock_detect_anomalies, mock_send_anomaly_alert,
        mock_aggregate_daily_products, mock_extracting_time_series_and_lagged_features,
        mock_save_cleaned_data, mock_upload_df_to_gcs
    ):

        # Test
        main(cloud=True)

        # Assert
        mock_load_bucket_data.assert_called_once()
        mock_filling_missing_dates.assert_called_once()
        mock_convert_feature_types.assert_called_once()
        mock_convert_string_columns_to_lowercase.assert_called_once()
        mock_standardize_product_name.assert_called_once()
        mock_apply_fuzzy_correction.assert_called_once()
        mock_filling_missing_cost_price.assert_called_once()
        mock_remove_invalid_records.assert_called_once()
        mock_remove_duplicate_records.assert_called_once()
        mock_detect_anomalies.assert_called_once()
        mock_send_anomaly_alert.assert_called_once()
        mock_aggregate_daily_products.assert_called_once()
        mock_extracting_time_series_and_lagged_features.assert_called_once()
        mock_upload_df_to_gcs.assert_called_once()

        mock_save_cleaned_data.assert_not_called()




class TestLoadBucketData(unittest.TestCase):
    def setUp(self):
        
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

    @patch('scripts.dataPreprocessing.storage.Client')
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

    @patch('scripts.dataPreprocessing.storage.Client')
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

    @patch('scripts.dataPreprocessing.storage.Client')
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

    @patch('scripts.dataPreprocessing.storage.Client')
    @patch('scripts.dataPreprocessing.pl.read_excel')
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

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_price_anomalies(self, mock_iqr_bounds):
        """Test detection of price anomalies"""
        mock_iqr_bounds.return_value = (1.0, 10.0)

        # Run the function
        anomalies, clean_df = detect_anomalies(self.test_data)
        
        # Check that price anomalies were detected
        self.assertIn('price_anomalies', anomalies)
        mock_iqr_bounds.assert_called()
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


    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_quantity_anomalies(self, mock_iqr_bounds):
        """Test detection of quantity anomalies"""
        mock_iqr_bounds.return_value = (2, 15)
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
    

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_time_anomalies(self, mock_iqr_bounds):
        """Test detection of time pattern anomalies"""
        mock_iqr_bounds.return_value = (10, 2)
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

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_format_anomalies(self, mock_iqr_bounds):
        """Test detection of format anomalies (invalid values)"""
        mock_iqr_bounds.return_value = None
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

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_empty_dataframe(self, mock_iqr_bounds):
        """Test function behavior with an empty dataframe"""
        mock_iqr_bounds.return_value = None
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

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_insufficient_data_for_iqr(self, mock_iqr_bounds):
        """Test function behavior with insufficient data points for IQR analysis"""

        mock_iqr_bounds.return_value = (1.0, 10.0)
        # Run the function with small data
        anomalies, clean_df = detect_anomalies(self.small_data)
        
        # Should not detect price or quantity anomalies due to insufficient data
        self.assertIn('price_anomalies', anomalies)
        self.assertEqual(len(anomalies['price_anomalies']), 0)
        
        self.assertIn('quantity_anomalies', anomalies)
        self.assertEqual(len(anomalies['quantity_anomalies']), 0)
        
        # Clean data should match original data (no anomalies removed)
        self.assertEqual(len(clean_df), len(self.small_data))

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_error_handling(self, mock_iqr_bounds):
        """Test error handling in the function"""
        mock_iqr_bounds.return_value = None
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

    @patch("scripts.dataPreprocessing.iqr_bounds")
    def test_clean_data_integrity(self, mock_iqr_bounds):
        """Test that the clean data maintains integrity of non-anomalous records"""
        mock_iqr_bounds.return_value = (1.0, 10.0)
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

    @patch('scripts.dataPreprocessing.storage')
    @patch('scripts.dataPreprocessing.logger')
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

    @patch('scripts.dataPreprocessing.storage')
    @patch('scripts.dataPreprocessing.logger')
    def test_google_cloud_error(self, mock_logging, mock_storage):
        # Configure mocks
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Setup chain of mock returns
        mock_storage.Client.return_value = mock_client
        mock_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Make the upload_from_string method raise a GoogleAPICallError
        cloud_error = GoogleAPICallError("Storage error")
        mock_blob.upload_from_string.side_effect = cloud_error
        
        # Call the function and check that it raises the error
        with self.assertRaises(GoogleAPICallError) as context:
            upload_df_to_gcs(self.test_df, self.bucket_name, self.blob_name)
        
        # Verify logging
        self.assertEqual("Storage Error", str(context.exception))
        # mock_logging.error.assert_called_with(
        #     "Error uploading DataFrame to GCS. Error: %s", cloud_error
        # )

    @patch('scripts.dataPreprocessing.storage')
    @patch('scripts.dataPreprocessing.logger')
    def test_general_exception(self, mock_logging, mock_storage):
        # Configure mocks
        mock_client = MagicMock()
        
        # Make get_bucket raise a general exception
        mock_client.get_bucket.side_effect = Exception("General error")
        mock_storage.Client.return_value = mock_client
        
        # Call the function and check that it raises the error
        with self.assertRaises(Exception):
            upload_df_to_gcs(self.test_df, self.bucket_name, self.blob_name)
        
        # Verify logging
        mock_logging.error.assert_called_with(
            "Error uploading DataFrame to GCS. Error: "
        )

    @patch('scripts.dataPreprocessing.storage')
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

class TestLoadBucketData(unittest.TestCase):
    def setUp(self):
        
        # Create mock data for tests
        self.mock_bucket_name = "test-bucket"
        self.mock_file_name = "test-file.xlsx"
        
        # Create a sample DataFrame for testing
        self.sample_df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    @patch('scripts.dataPreprocessing.storage.Client')
    @patch('scripts.dataPreprocessing.pl.read_excel')
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

    @patch('scripts.dataPreprocessing.storage.Client')
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

    @patch('scripts.dataPreprocessing.storage.Client')
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

    @patch('scripts.dataPreprocessing.storage.Client')
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

    @patch('scripts.dataPreprocessing.storage.Client')
    @patch('scripts.dataPreprocessing.pl.read_excel')
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

    @patch('scripts.dataPreprocessing.storage')
    @patch('scripts.dataPreprocessing.logger')
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

    @patch('scripts.dataPreprocessing.storage')
    @patch('scripts.dataPreprocessing.logger')
    def test_google_cloud_error(self, mock_logging, mock_storage):
        # Configure mocks
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        # Setup chain of mock returns
        mock_storage.Client.return_value = mock_client
        mock_client.get_bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Make the upload_from_string method raise a GoogleAPICallError
        cloud_error = GoogleAPICallError("Storage error")
        mock_blob.upload_from_string.side_effect = cloud_error
        
        # Call the function and check that it raises the error
        with self.assertRaises(GoogleAPICallError):
            upload_df_to_gcs(self.test_df, self.bucket_name, self.blob_name)
        
        # Verify logging
        mock_logging.error.assert_called_with(
            "Error uploading DataFrame to GCS. Error: %s", cloud_error
        )

    @patch('scripts.dataPreprocessing.storage')
    @patch('scripts.dataPreprocessing.logger')
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

    @patch('scripts.dataPreprocessing.storage')
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
    

