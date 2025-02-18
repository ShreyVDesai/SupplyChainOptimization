import numpy as np
import pandas as pd
import polars as pl

def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pl.read_excel(file_path)


def convert_feature_types(df):
    """Converts columns to their appropriate data types for consistency.
    
    - Converts 'Date' to `Datetime`
    - Converts numeric features to `Float64` or `Int64`
    - Ensures categorical features remain as `Utf8`

    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Updated DataFrame with corrected types
    """
    
    # Define expected data types
    expected_dtypes = {
        "Date": pl.Datetime,
        "Cost Price": pl.Float64,
        "Quantity": pl.Int64, 
        "Total Price": pl.Float64, 
        "Transaction ID": pl.Utf8,
        "Store Location": pl.Utf8,
        "Product Name": pl.Utf8,
        "Producer ID": pl.Utf8
    }
    
    # Convert columns to expected data types
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    
    return df


def fill_missing_value_with_unknown(df, features):
    """Fills missing values in specified columns with the string 'Unknown'.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
        features (list): List of feature names to process

    Returns:
        pl.DataFrame: Updated DataFrame with missing values replaced
    """
    for feature in features:
        df = df.with_columns(
            pl.col(feature)
            .cast(pl.Utf8)
            .fill_null("Unknown")
        )

    return df


def convert_string_columns_to_lowercase(df):
    """Converts all string-type columns to lowercase for consistency.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Updated DataFrame with lowercase strings
    """
    # Detect all string-type columns dynamically
    string_features = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]

    df = df.with_columns(
        [pl.col(feature).str.to_lowercase() for feature in string_features]
    )

    return df




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
    # df = df.with_columns(pl.col("Date").fill_null(strategy="forward"))
    # df = df.with_columns(pl.col("Date").fill_null(strategy="backward"))

    # df = df.sort("Date")

    return df



def extract_datetime_features(df):
    """
    Extract Year, Month and Week of year from date column.
    """
    return df.with_columns(
        pl.col("Date").dt.year().alias("Year"),
        pl.col("Date").dt.month().alias("Month"),
        pl.col("Date").dt.week().alias("Week_of_year")
    )


def compute_most_frequent_price(df, time_granularity):
    """
    Computes the most frequent cost price for each product at different time granularities.
    
    Parameters:
        df (pl.DataFrame): Input dataframe
        time_granularity (list): List of time-based features for grouping (e.g., ["Year", "Month", "Week"])
    
    Returns:
        pl.DataFrame: Mapping of (time + product) → most frequent cost price
    """

    return (
        df.drop_nulls(["Cost Price"])
        .group_by(time_granularity + ["Product Name"])
        .agg(pl.col("Cost Price").mode().first().alias("Most_Frequent_Cost"))
    )


def filling_missing_cost_price(df):
    """
    Fills missing 'Cost Price' values based on the most frequent price at different time granularities.

    Parameters:
        df (pl.DataFrame): Input dataframe

    Returns:
        pl.DataFrame: Updated dataframe with missing cost prices filled.
    """
    # Extract time features dynamically
    df = extract_datetime_features(df)

    # Compute most frequent prices at different time levels
    price_by_week = compute_most_frequent_price(df, ["Year", "Month", "Week_of_year"])
    price_by_month = compute_most_frequent_price(df, ["Year", "Month"])
    price_by_year = compute_most_frequent_price(df, ["Year"])
    

    # Merge with original dataframe to fill missing values
    df = df.join(price_by_week, on=["Year", "Month", "Week_of_year", "Product Name"], how="left")
    df = df.with_columns(
        pl.when(pl.col("Cost Price").is_null())
        .then(pl.col("Most_Frequent_Cost"))
        .otherwise(pl.col("Cost Price"))
        .alias("Cost Price")
    ).drop("Most_Frequent_Cost")


    df = df.join(price_by_month, on=["Year", "Month", "Product Name"], how="left")
    df = df.with_columns(
        pl.when(pl.col("Cost Price").is_null())
        .then(pl.col("Most_Frequent_Cost"))
        .otherwise(pl.col("Cost Price"))
        .alias("Cost Price")
    ).drop("Most_Frequent_Cost")


    df = df.join(price_by_year, on=["Year", "Product Name"], how="left")
    df = df.with_columns(
        pl.when(pl.col("Cost Price").is_null())
        .then(pl.col("Most_Frequent_Cost"))
        .otherwise(pl.col("Cost Price"))
        .alias("Cost Price")
    ).drop("Most_Frequent_Cost")


    # If still null, set to "Unknown" or a default value (e.g., 0)
    df = df.with_columns(
        pl.col("Cost Price").fill_null(0)
    )

    return df



# If Quantity or Product Name is missing, remove those records.
def remove_invalid_records(df):
    """
    Remove records if Quantity or Product Name has invalid input.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Filtered DataFrame with invalid records removed.
    """
    
    return df.filter(
        (pl.col("Quantity") > 0) &
        pl.col("Quantity").is_not_null() &
        pl.col("Product Name").is_not_null()
    )




# def filter_valid_records(df, col_names):
#     """
#     Filters out invalid records useful for demand forecasting.
#     Ensures values are not null and greater than 0 for multiple columns.

#     Parameters:
#         df (pl.DataFrame): Input Polars DataFrame
#         col_names (list): List of column names to filter

#     Returns:
#         pl.DataFrame: Filtered DataFrame
#     """
#     for col_name in col_names:
#         df = df.filter(pl.col(col_name).is_not_null() & (pl.col(col_name) > 0))
#     return df



def save_cleaned_data(df, output_file):
    """Saves the cleaned data to a CSV file."""
    df.write_csv(output_file)



def main(input_file, output_file):
    """Executes all cleaning steps."""
    df = load_data(input_file)

    df = clean_dates(df)

    df = convert_feature_types(df)

    df = fill_missing_value_with_unknown(df, ["Producer ID", "Store Location", "Transaction ID"])

    # df = replace_digits_in_string_columns(df)

    df = convert_string_columns_to_lowercase(df)

    df = filling_missing_cost_price(df)

    df = remove_invalid_records(df)
    
    # print("Cleaning Store Location Data...")
    # df = clean_store_location(df)
    
    # print("Cleaning Product Name Data...")
    # df = clean_product_names(df)
    
    # print("Filtering Valid Records...")
    # df = filter_valid_records(df, ["Quantity", "Cost Price"])
    
    # print("Cleaning Date Data...")
    # df = clean_dates(df)
    
    print("Saving Cleaned Data...")
    save_cleaned_data(df, output_file)
    
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    input_file = "messy_transactions_20190103_20241231.xlsx"  # Change to actual file
    output_file = "cleaned_data.csv"
    main(input_file, output_file)
