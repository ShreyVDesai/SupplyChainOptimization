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


def calculate_total_price(df):
    """Calculating total price of the transaction"""
    df = df.with_columns(
        (pl.col("Cost Price") * pl.col("Quantity")).alias("Total Price")
    )

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
    df = filter_valid_records(df, ["Quantity", "Cost Price", "Producer ID"])
    
    print("Cleaning Date Data...")
    df = clean_dates(df)

    print("Calculating Total Price...")
    df = calculate_total_price(df)
    
    print("Saving Cleaned Data...")
    save_cleaned_data(df, output_file)
    
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    input_file = "messy_transactions_20190103_20241231.xlsx"
    output_file = "cleaned_data.csv"
    main(input_file, output_file)