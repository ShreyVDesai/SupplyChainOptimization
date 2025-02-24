import polars as pl
from polars.testing import assert_frame_equal
from datetime import datetime
import io
import unittest
from unittest.mock import MagicMock, patch
from scripts.dataPreprocessing import convert_feature_types, compute_most_frequent_price, standardize_date_format, detect_date_order, filling_missing_dates, remove_future_dates, extract_datetime_features

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


    # Test case where it is a valid input.
    def test_convert_types_valid_df(self):
        """Test if the function correctly converts valid data types."""
        df_converted = convert_feature_types(self.valid_df)

        self.assertEqual(df_converted.schema["Date"], pl.Datetime, "Date column is not converted to Datetime")
        self.assertEqual(df_converted.schema["Unit Price"], pl.Float64, "Unit Price is not Float64")
        self.assertEqual(df_converted.schema["Quantity"], pl.Int64, "Quantity is not Int64")
        self.assertEqual(df_converted.schema["Transaction ID"], pl.Utf8, "Transaction ID is not Utf8")
        self.assertEqual(df_converted.schema["Store Location"], pl.Utf8, "Store Location is not Utf8")
        self.assertEqual(df_converted.schema["Product Name"], pl.Utf8, "Product Name is not Utf8")
        self.assertEqual(df_converted.schema["Producer ID"], pl.Utf8, "Producer ID is not Utf8")



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

    # Test case where "Date" column not found.
    def test_detect_date_order_missing_date_col(self):
        # Setup
        df = pl.DataFrame({"NotDate": ["2021-01-01", "2021-01-02"]})
        df = standardize_date_format(df)

        # Test
        with self.assertRaises(AttributeError) as context:
            detect_date_order(df, "Date")

            # Assert
            self.assertIn("Date column not found.", context.exception)


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
    
    
    # Test case where Ascending order with missing dates.
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

        expected_dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
        
        # Test
        result_df = filling_missing_dates(df, "Date")
        result_df = self.convert_dates_to_str(result_df, "Date")
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_dates, expected_dates)



    # Test case where Descending order with missing dates.
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

        expected_dates = ["2020-01-05", "2020-01-04", "2020-01-03", "2020-01-02", "2020-01-01"]
        
        # Test
        result_df = filling_missing_dates(df, "Date")
        result_df = self.convert_dates_to_str(result_df, "Date")
        result_dates = result_df["Date"].to_list()

        # Assert
        self.assertEqual(result_dates, expected_dates)



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

    



if __name__ == "__main__":
    unittest.main()

    
        