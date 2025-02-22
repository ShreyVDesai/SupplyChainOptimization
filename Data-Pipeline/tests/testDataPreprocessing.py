import polars as pl
import unittest
from scripts.dataPreprocessing import convert_feature_types

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.valid_df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Cost Price": [100.5, 200.75],
            "Quantity": [10, 20],
            "Transaction ID": ["T123", "T456"],
            "Store Location": ["NY", "CA"],
            "Product Name": ["milk", "coffee"],
            "Producer ID": ["P001", "P002"]
        }).with_columns(pl.col("Date").str.to_datetime())

        # Missing features
        self.missing_columns_df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Cost Price": [100.5, 200.75],
            "Quantity": [10, 20]
        })

    # Test case where it is a valid input.
    def test_convert_types_valid_df(self):
        """Test if the function correctly converts valid data types."""
        df_converted = convert_feature_types(self.valid_df)

        self.assertEqual(df_converted.schema["Date"], pl.Datetime, "Date column is not converted to Datetime")
        self.assertEqual(df_converted.schema["Cost Price"], pl.Float64, "Cost Price is not Float64")
        self.assertEqual(df_converted.schema["Quantity"], pl.Int64, "Quantity is not Int64")
        self.assertEqual(df_converted.schema["Transaction ID"], pl.Utf8, "Transaction ID is not Utf8")
        self.assertEqual(df_converted.schema["Store Location"], pl.Utf8, "Store Location is not Utf8")
        self.assertEqual(df_converted.schema["Product Name"], pl.Utf8, "Product Name is not Utf8")
        self.assertEqual(df_converted.schema["Producer ID"], pl.Utf8, "Producer ID is not Utf8")

    
    # Test case where features are missing.
    def test_convert_types_missing_features_df(self):
        """Test if the function correctly converts valid data types."""

        with self.assertRaises(KeyError) as context:
            convert_feature_types(self.missing_columns_df)
        self.assertIn("Missing columns in DataFrame: ['Transaction ID', 'Store Location', 'Product Name', 'Producer ID']", str(context.exception))



if __name__ == "__main__":
    unittest.main()

    
        