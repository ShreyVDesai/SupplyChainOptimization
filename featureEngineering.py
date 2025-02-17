import polars as pl
import pandas as pd


def load_data(file_path):
    """Loads the dataset from the given file path."""
    return pl.read_csv(file_path)


def clean_date(df):
    """
    Standardizes date formats in the 'Date' column to '%Y-%m-%d %H:%M:%S.%f'.
    - Removes 'T' from ISO timestamps (e.g., 2019-01-03T00:00:00.000 → 2019-01-03 00:00:00.000)
    - Ensures all dates are correctly parsed as datetime.
    """

    # Ensure Date is a string before processing
    df = df.with_columns(pl.col("Date").cast(pl.Utf8))

    # Remove 'T' from ISO timestamps & convert to datetime
    df = df.with_columns(
        pl.col("Date")
        .str.replace("T", " ")  # Convert "T" to space
        .str.to_datetime(format="%Y-%m-%d %H:%M:%S.%3f", strict=False)  # Convert to Datetime
        .alias("Date")
    )

    return df


def extract_time_features(df):
    """
    Extracts time-based features from the Date column.
    """

    df = df.with_columns(
        pl.col("Date").dt.day().alias("Day"),
        pl.col("Date").dt.month().alias("Month"),
        pl.col("Date").dt.year().alias("Year"),
        pl.col("Date").dt.hour().alias("Hour"),
        pl.col("Date").dt.weekday().alias("Weekday"),
        pl.col("Date").dt.week().alias("Week_of_year"),
        pl.col("Date").dt.quarter().alias("Quarter"),
        (pl.col("Date").dt.weekday().is_in([5, 6])).cast(pl.Int8).alias("Is_Weekend")
    )

    return df


def aggregate_daily_sales(df):
    """Aggregates the data at a daily level by store and product."""

    aggregated_df = (
        df.group_by(["Year", "Month", "Day", "Store Location", "Product Name", "Producer ID"])
        .agg([
            pl.len().alias("Transactions_Per_Day"),
            pl.sum("Quantity").alias("Total_Quantity_Sold"), # Total Products sold
            # pl.std("Quantity").alias("Std_Dev_Quantity"), # Variability in product demand
            pl.sum("Total Price").alias("Total_Revenue"), # Total Revenue
        ])
    )


    # Compute 7-day rolling avg grouped by Store & Product
    aggregated_df = aggregated_df.with_columns(
        pl.col("Total_Quantity_Sold")
        .rolling_mean(window_size=7)
        .over(["Store Location", "Product Name", "Producer ID"])
        .alias("Rolling_Avg_Sales")  # 7-Day moving average of sales
    )

    aggregated_df = aggregated_df.sort(["Year", "Month", "Day"])


    return aggregated_df


def feature_engineering_pipeline(df):
    """Pipeline for feature extraction and aggregation."""
    
    df = clean_date(df)
    df = extract_time_features(df)
    daily_aggregated_df = aggregate_daily_sales(df)

    return daily_aggregated_df


def save_aggregated_data(df, file_name):
    """Saves the aggregated dataframe to an Excel file."""
    df.write_csv(file_name)
    print(f"Aggregated data saved to {file_name}")


if __name__ == "__main__":
    input_file = "cleaned_data.csv"
    output_file = "aggregated_data.csv"
    df = load_data(input_file)
    df = feature_engineering_pipeline(df)
    save_aggregated_data(df, output_file)


