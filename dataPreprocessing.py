import numpy as np
import pandas as pd
import polars as pl

def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pl.read_excel(file_path)


def clean_dates(df):
    """
    Standardizes date formats in the 'Date' column to a single format: '%Y-%m-%d %H:%M:%S.%f'.

    - Converts multiple formats into a single standard format.
    - Preserves timestamps if already present.
    - Adds '00:00:00.000' for dates missing time.
    - Ensures uniform datetime format.
    """

    # Ensure 'Date' column is a string before processing
    df = df.with_columns(pl.col("Date").cast(pl.Utf8))

    # Convert different date formats to a consistent datetime format
    df = df.with_columns(
        pl.when(pl.col("Date").str.contains(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$"))  # 2019-01-03 08:46:08
        .then(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f", strict=False))

        .when(pl.col("Date").str.contains(r"^\d{4}-\d{2}-\d{2}$"))  # 2019-01-03
        .then(pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False))

        .when(pl.col("Date").str.contains(r"^\d{2}-\d{2}-\d{4}$"))  # 01-03-2019
        .then(pl.col("Date").str.strptime(pl.Datetime, "%m-%d-%Y", strict=False))

        .when(pl.col("Date").str.contains(r"^\d{2}/\d{2}/\d{4}$"))  # 03/01/2019
        .then(pl.col("Date").str.strptime(pl.Datetime, "%d/%m/%Y", strict=False))

        .otherwise(None)  # Handle unknown formats
        .alias("Date")
    )

    # Fill missing dates using forward-fill and backward-fill
    df = df.with_columns(pl.col("Date").fill_null(strategy="forward"))
    df = df.with_columns(pl.col("Date").fill_null(strategy="backward"))

    # df = df.sort("Date")

    return df



def clean_store_location(df):
    """Fixes store location inconsistencies and removes invalid entries."""
    df = df.with_columns(
        pl.col("Store Location")
        .cast(pl.Utf8)
        .fill_null("Unknown")
        .str.to_lowercase()
        .map_elements(lambda x: "unknown" if x.isdigit() else x)
    )

    # Remove invalid store locations
    df = df.filter(
        (pl.col("Store Location") != "unknown") & 
        (pl.col("Store Location") != "nan") & 
        (pl.col("Store Location").is_not_null())
    )

    return df

def clean_product_names(df):
    """Handles typos and missing values in product names."""
    df = df.with_columns(
        pl.col("Product Name")
        .fill_null("Unknown")
        .str.to_lowercase()
        .str.replace_all("@", "a")
        .str.replace_all("0", "o")
    )

    # Remove unknown values
    df = df.filter(
        (pl.col("Product Name") != "unknown") & 
        (pl.col("Product Name") != "nan") & 
        (pl.col("Product Name").is_not_null())
    )

    return df

def filter_valid_records(df, col_names):
    """
    Filters out invalid records useful for demand forecasting.
    Ensures values are not null and greater than 0 for multiple columns.

    Parameters:
        df (pl.DataFrame): Input Polars DataFrame
        col_names (list): List of column names to filter

    Returns:
        pl.DataFrame: Filtered DataFrame
    """
    for col_name in col_names:
        df = df.filter(pl.col(col_name).is_not_null() & (pl.col(col_name) > 0))
    return df



def save_cleaned_data(df, output_file):
    """Saves the cleaned data to a CSV file."""
    df.write_csv(output_file)

def main(input_file, output_file):
    """Executes all cleaning steps."""
    df = load_data(input_file)
    
    print("Cleaning Store Location Data...")
    df = clean_store_location(df)
    
    print("Cleaning Product Name Data...")
    df = clean_product_names(df)
    
    print("Filtering Valid Records...")
    df = filter_valid_records(df, ["Quantity", "Cost Price"])
    
    print("Cleaning Date Data...")
    df = clean_dates(df)
    
    print("Saving Cleaned Data...")
    save_cleaned_data(df, output_file)
    
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    input_file = "messy_transactions_20190103_20241231.xlsx"  # Change to actual file
    output_file = "cleaned_data.csv"
    main(input_file, output_file)


# def validate_data_types(df):
#     expected_data_types = {
#         'TransactionID': str,
#         'Date': 'datetime64[ns]',
#         'CostPrice': float,
#         'Quantity': int,
#         'ProductID': str,
#         'Store Location': str
#     }

#     for col, expected_type in expected_data_types.items():
#         if col in df.columns:
#             if expected_type == 'datetime64[ns]':
#                 df[col] = pd.to_datetime(df[col], errors='coerce')
#             else:
#                 df[col] = df[col].astype(expected_type)

#     return df


# def remove_empty_records(df):
#     '''Removes records where either Quantity or ProductID are empty.'''
#     return df.dropna(subset=['Quantity', 'ProductID'], how='any')

# def impute_cost_price(df):
#     pass

# def convert_date_format(df):
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     return df

# def extract_date_features(df):
#     df['Year'] = df['Date'].dt.year
#     df['Month'] = df['Date'].dt.month
#     df['Day'] = df['Date'].dt.day
#     df['DayOfWeek'] = df['Date'].dt.day_name()
#     df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
#     return df


# def remove_duplicates(df):
#     return df.drop_duplicates(subset=['TransactionID'])


# def standardize_categorical_features(df):
#     df['Store Location'] = df['Store Location'].str.strip().str.title()
#     df['ProductID'] = df['ProductID'].str.strip().str.upper()
#     return df


# def preprocess_data(file_path, save_path):
#     df = load_data(file_path)
#     # df = validate_data_types(df)
#     df = remove_empty_records(df)
#     df = convert_date_format(df)
#     df = extract_date_features(df)
#     df = remove_duplicates(df)
#     df = standardize_categorical_features(df)
#     df.to_excel(save_path, index=False)
#     print(f"Data Preprocessing Completed and file is stored at {save_path}")


# input_file = 'synthetic_transaction_data.xlsx'
# output_file = 'processed_transaction_data.xlsx'
# preprocess_data(input_file, output_file)