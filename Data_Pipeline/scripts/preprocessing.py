from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

try:
    from logger import logger
    from post_validation import post_validation
    from utils import (
        delete_blob_from_bucket,
        list_bucket_blobs,
        load_bucket_data,
        send_anomaly_alert,
        upload_to_gcs,
    )
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger
    from Data_Pipeline.scripts.utils import (
        load_bucket_data,
        upload_to_gcs,
        list_bucket_blobs,
        delete_blob_from_bucket,
        send_anomaly_alert,
    )
    from Data_Pipeline.scripts.post_validation import post_validation

import argparse
import os
from typing import Dict, Tuple

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# Reference product names (correct names from the dataset)
REFERENCE_PRODUCT_NAMES = [
    "milk",
    "coffee",
    "wheat",
    "chocolate",
    "beef",
    "sugar",
    "corn",
    "soybeans",
]


def convert_feature_types(df: pl.DataFrame) -> pl.DataFrame:
    """
    Converts columns to their appropriate data types for consistency:
    - 'Date' → Datetime
    - 'Unit Price' → Float64
    - 'Quantity' → Int64
    - Others (categoricals) → Utf8
    """
    expected_dtypes = {
        "Date": pl.Datetime,
        "Unit Price": pl.Float64,
        "Quantity": pl.Int64,
        "Transaction ID": pl.Utf8,
        "Store Location": pl.Utf8,
        "Product Name": pl.Utf8,
        "Producer ID": pl.Utf8,
    }
    try:
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                # For Quantity, first convert to float and round to integer
                if col == "Quantity":
                    df = df.with_columns(
                        pl.col(col).cast(pl.Float64).round(0).cast(pl.Int64)
                    )
                else:
                    df = df.with_columns(pl.col(col).cast(dtype))
        logger.info("Feature types converted successfully.")
        return df
    except Exception as e:
        logger.error(
            "Unexpected error during feature type conversion.", exc_info=True
        )
        raise


def convert_string_columns_to_lowercase(df: pl.DataFrame) -> pl.DataFrame:
    """Converts all string-type columns to lowercase for consistency."""
    try:
        logger.info("Converting all string-type columns to lowercase...")
        string_features = [
            col
            for col, dtype in zip(df.columns, df.dtypes)
            if dtype == pl.Utf8
        ]
        if not string_features:
            logger.warning("No string columns detected. Skipping conversion.")
            return df

        df = df.with_columns(
            [pl.col(feature).str.to_lowercase() for feature in string_features]
        )
        return df
    except Exception as e:
        logger.error(f"Unexpected error during string conversion: {e}")
        raise


def standardize_date_format(
    df: pl.DataFrame, date_column: str = "Date"
) -> pl.DataFrame:
    """
    Standardizes date formats in the given column, handling multiple
    date string formats and casting them to a consistent datetime type.
    """
    try:
        logger.info("Standardizing date formats...")

        if date_column not in df.columns:
            logger.error(f"Column '{date_column}' not found in DataFrame.")
            return df

        # Make sure we're working with string representations first
        df = df.with_columns(pl.col(date_column).cast(pl.Utf8))

        # Handle empty DataFrame
        if df.is_empty():
            return df

        # Create a more comprehensive date conversion
        df = df.with_columns(
            pl.when(
                pl.col(date_column).str.contains(
                    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$"
                )
            )  # 2019-01-03 08:46:08
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{4}-\d{2}-\d{2}$")
            )  # 2019-01-03
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%Y-%m-%d", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{2}-\d{2}-\d{4}$")
            )  # 01-03-2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%m-%d-%Y", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{2}/\d{2}/\d{4}$")
            )  # 03/01/2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%d/%m/%Y", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{1,2}/\d{1,2}/\d{4}$")
            )  # 3/1/2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%m/%d/%Y", strict=False
                )
            )
            .when(
                pl.col(date_column).str.contains(r"^\d{1,2}-\d{1,2}-\d{4}$")
            )  # 3-1-2019
            .then(
                pl.col(date_column).str.strptime(
                    pl.Datetime, "%m-%d-%Y", strict=False
                )
            )
            .otherwise(None)
            .alias(date_column)
        )

        # Try to convert any null values with additional formats
        null_mask = df[date_column].is_null()
        if null_mask.sum() > 0:
            logger.warning(
                f"Some date values could not be parsed: {null_mask.sum()} nulls"
            )

            # Try one more time with the pandas parser which is more flexible
            try:
                # Convert to pandas, apply conversion, then back to polars
                temp_df = df.filter(null_mask).to_pandas()
                if not temp_df.empty:
                    temp_df[date_column] = pd.to_datetime(
                        temp_df[date_column], errors="coerce"
                    )
                    temp_pl = pl.from_pandas(temp_df)

                    # Update only the previously null values
                    df = df.with_columns(
                        pl.when(null_mask)
                        .then(pl.lit(temp_pl[date_column]))
                        .otherwise(pl.col(date_column))
                        .alias(date_column)
                    )
            except Exception as e:
                logger.warning(f"Additional date parsing attempt failed: {e}")

        null_count = df[date_column].null_count()
        if null_count > 0:
            logger.warning(
                f"Date column '{date_column}' has {null_count} null values after conversion. Check data format."
            )

        logger.info("Date standardization completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Unexpected error during date processing: {e}")
        # Return original DataFrame rather than failing
        return df


def detect_date_order(df: pl.DataFrame, date_column: str = "Date") -> str:
    """
    Attempts to identify if the 'Date' column is in ascending, descending,
    or random order, ignoring null values.
    """
    try:
        df = df.with_columns(pl.col(date_column).cast(pl.Datetime))
        temp_df = df.with_columns(
            pl.col(date_column).dt.date().alias(date_column)
        )
        date_series = temp_df.drop_nulls(date_column)[date_column].to_list()

        if len(date_series) < 2:
            return "Random"

        is_ascending = all(
            date_series[i] <= date_series[i + 1]
            for i in range(len(date_series) - 1)
        )
        is_descending = all(
            date_series[i] >= date_series[i + 1]
            for i in range(len(date_series) - 1)
        )

        if is_ascending:
            return "Ascending"
        elif is_descending:
            return "Descending"
        else:
            return "Random"
    except Exception as e:
        logger.error(f"Error detecting date order: {e}")
        raise


def filling_missing_dates(
    df: pl.DataFrame, date_column: str = "Date"
) -> pl.DataFrame:
    """
    Fills missing dates using forward/backward fill, or drops them if no clear order is found.
    """
    try:
        df = standardize_date_format(df, date_column)

        if df[date_column].null_count() > 0:
            order_type = detect_date_order(df, date_column)
            logger.warning(
                f"{df[date_column].null_count()} missing date values before filling."
            )

            df = df.with_columns(
                [
                    pl.col(date_column)
                    .fill_null(strategy="forward")
                    .alias("prev_valid"),
                    pl.col(date_column)
                    .fill_null(strategy="backward")
                    .alias("next_valid"),
                ]
            )
            original_count = df.height

            # Drop rows whose date can't be consistently interpolated
            df = df.filter(
                pl.col(date_column).is_not_null()
                | (
                    (
                        pl.col("prev_valid").dt.truncate("1d")
                        == pl.col("next_valid").dt.truncate("1d")
                    )
                )
            )

            dropped_count = original_count - df.height
            logger.info(
                f"Dropped {dropped_count} records due to mismatched date boundaries."
            )

            df = df.drop(["prev_valid", "next_valid"])

            if order_type == "Ascending":
                logger.info("Ascending Order Detected: Using Forward-Fill")
                df = df.with_columns(pl.col(date_column).interpolate())
            elif order_type == "Descending":
                logger.info("Descending Order Detected: Using Backward-Fill")
                df = df.with_columns(pl.col(date_column).interpolate())
            else:
                logger.warning("Random Order Detected: Dropping Missing Dates")
                df = df.filter(pl.col(date_column).is_not_null())

        else:
            logger.info("No null values found in Date feature.")
        return df
    except Exception as e:
        logger.error(f"Error filling missing dates: {e}")
        raise


def extract_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extracts Year, Month, and Week_of_year from the 'Date' column."""
    try:
        return df.with_columns(
            pl.col("Date").dt.year().alias("Year"),
            pl.col("Date").dt.month().alias("Month"),
            pl.col("Date").dt.week().alias("Week_of_year"),
        )
    except Exception as e:
        logger.error("Error extracting datetime features.")
        raise


def compute_most_frequent_price(
    df: pl.DataFrame, time_granularity: list
) -> pl.DataFrame:
    """
    Computes the most frequent (mode) unit price for each product
    at different time granularities (Year, Month, Week, etc.).
    """
    try:
        return (
            df.drop_nulls(["Unit Price"])
            .group_by(time_granularity + ["Product Name"])
            .agg(
                pl.col("Unit Price").mode().first().alias("Most_Frequent_Cost")
            )
        )
    except Exception as e:
        logger.error(f"Error computing most frequent unit price: {e}")
        raise


def filling_missing_cost_price(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fills missing 'Unit Price' by sequentially checking the most
    frequent price at (Year, Month, Week_of_year), then (Year, Month),
    then (Year). Remaining nulls are filled with 0.
    """
    try:
        logger.info("Filling missing Unit Price values...")
        df = extract_datetime_features(df)
        price_by_week = compute_most_frequent_price(
            df, ["Year", "Month", "Week_of_year"]
        )
        price_by_month = compute_most_frequent_price(df, ["Year", "Month"])
        price_by_year = compute_most_frequent_price(df, ["Year"])

        # Merge each level of granularity
        df = df.join(
            price_by_week,
            on=["Year", "Month", "Week_of_year", "Product Name"],
            how="left",
        )
        df = df.with_columns(
            pl.when(pl.col("Unit Price").is_null())
            .then(pl.col("Most_Frequent_Cost"))
            .otherwise(pl.col("Unit Price"))
            .alias("Unit Price")
        ).drop("Most_Frequent_Cost")

        df = df.join(
            price_by_month, on=["Year", "Month", "Product Name"], how="left"
        )
        df = df.with_columns(
            pl.when(pl.col("Unit Price").is_null())
            .then(pl.col("Most_Frequent_Cost"))
            .otherwise(pl.col("Unit Price"))
            .alias("Unit Price")
        ).drop("Most_Frequent_Cost")

        df = df.join(price_by_year, on=["Year", "Product Name"], how="left")
        df = df.with_columns(
            pl.when(pl.col("Unit Price").is_null())
            .then(pl.col("Most_Frequent_Cost"))
            .otherwise(pl.col("Unit Price"))
            .alias("Unit Price")
        ).drop("Most_Frequent_Cost")

        # Fill remaining nulls with 0
        df = df.with_columns(pl.col("Unit Price").fill_null(0))
        logger.info("Unit Price filling completed successfully.")

        df = df.drop(["Year", "Month", "Week_of_year"])
        return df
    except Exception as e:
        logger.error(
            f"Unexpected error while filling missing Unit Prices: {e}"
        )
        raise


def remove_invalid_records(df: pl.DataFrame) -> pl.DataFrame:
    """
    Removes records if Quantity or Product Name is null.
    """
    try:
        return df.filter(
            pl.col("Quantity").is_not_null()
            & pl.col("Product Name").is_not_null()
        )
    except Exception as e:
        logger.error(f"Error removing invalid records: {e}")
        raise


def standardize_product_name(df: pl.DataFrame) -> pl.DataFrame:
    """
    Removes extraneous characters and converts product names to lowercase.
    """
    try:
        logger.info("Standardizing product name.")
        df = df.with_columns(
            pl.col("Product Name")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_lowercase()
            .alias("Product Name")
        )
        return df
    except Exception as e:
        logger.error(f"Error standardizing product name: {e}")
        raise


def filter_invalid_products(
    df: pl.DataFrame, reference_list: list
) -> pl.DataFrame:
    """
    Filters out rows where the product name is not in the reference list.
    """
    try:
        logger.info("Filtering out invalid product names...")
        original_count = df.height
        df = df.filter(pl.col("Product Name").is_in(reference_list))
        filtered_count = original_count - df.height
        logger.info(
            f"Filtered out {filtered_count} rows with invalid product names."
        )
        return df
    except Exception as e:
        logger.error(f"Error filtering invalid products: {e}")
        raise


def calculate_zscore(series: pl.Series) -> pl.Series:
    """Calculate Z-score for a Polars Series."""
    try:
        mean = series.mean()
        std = series.std()
        if std == 0 or std is None:
            return pl.Series([0] * len(series))
        return (series - mean) / std
    except Exception as e:
        logger.error(f"Error calculating Z-score: {e}")
        raise


def iqr_bounds(series: pl.Series) -> Tuple[float, float]:
    """Calculate IQR-based lower and upper bounds for anomaly detection."""
    try:
        if series.is_empty():
            raise ValueError("Cannot compute IQR for an empty series.")

        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1

        if iqr == 0:
            return q1, q3

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (2.0 * iqr)
        if series.min() >= 0:
            lower_bound = max(0, lower_bound)

        logger.debug(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        return lower_bound, upper_bound
    except Exception as e:
        logger.error(f"Error calculating IQR bounds: {e}")
        raise


def detect_anomalies(
    df: pl.DataFrame,
) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame]:
    """
    Detect anomalies in transaction data per product per day using IQR checks,
    odd time-of-day checks, and invalid-format checks.
    Returns:
      - Dictionary of anomaly types → DataFrames
      - Clean DataFrame with anomalies removed
    """
    anomalies = {}
    clean_df = df.clone()
    anomaly_transaction_ids = set()

    try:
        df = df.with_columns(
            [
                pl.col("Date").cast(pl.Datetime).alias("datetime"),
                pl.col("Date").cast(pl.Datetime).dt.date().alias("date_only"),
                pl.col("Date").cast(pl.Datetime).dt.hour().alias("hour"),
            ]
        )

        # 1. Price Anomalies
        price_anomalies = []
        product_date_combinations = df.select(
            ["Product Name", "date_only"]
        ).unique()

        for row in product_date_combinations.iter_rows(named=True):
            product = row["Product Name"]
            date = row["date_only"]
            subset = df.filter(
                (pl.col("Product Name") == product)
                & (pl.col("date_only") == date)
            )

            if "Unit Price" in df.columns:
                if len(subset) >= 4:
                    lower_bound, upper_bound = iqr_bounds(subset["Unit Price"])
                    iqr_anoms = subset.filter(
                        (pl.col("Unit Price") < lower_bound)
                        | (pl.col("Unit Price") > upper_bound)
                    )
                    if len(iqr_anoms) > 0:
                        price_anomalies.append(iqr_anoms)
                        anomaly_transaction_ids.update(
                            iqr_anoms["Transaction ID"].to_list()
                        )

        anomalies["price_anomalies"] = (
            pl.concat(price_anomalies) if price_anomalies else pl.DataFrame()
        )
        logger.debug(f"Price anomalies detected: {len(price_anomalies)} sets.")

        # 2. Quantity Anomalies
        quantity_anomalies = []
        for row in product_date_combinations.iter_rows(named=True):
            product = row["Product Name"]
            date = row["date_only"]
            subset = df.filter(
                (pl.col("Product Name") == product)
                & (pl.col("date_only") == date)
            )

            if len(subset) >= 4:
                lower_bound, upper_bound = iqr_bounds(subset["Quantity"])
                iqr_anoms = subset.filter(
                    (pl.col("Quantity") < lower_bound)
                    | (pl.col("Quantity") > upper_bound)
                )
                if len(iqr_anoms) > 0:
                    quantity_anomalies.append(iqr_anoms)
                    anomaly_transaction_ids.update(
                        iqr_anoms["Transaction ID"].to_list()
                    )

        anomalies["quantity_anomalies"] = (
            pl.concat(quantity_anomalies)
            if quantity_anomalies
            else pl.DataFrame()
        )
        logger.debug(
            f"Quantity anomalies detected: {len(quantity_anomalies)} sets."
        )

        # 3. Time-of-day Anomalies
        time_anomalies = df.filter(
            (pl.col("hour") < 6) | (pl.col("hour") > 22)
        )
        anomalies["time_anomalies"] = time_anomalies
        anomaly_transaction_ids.update(
            time_anomalies["Transaction ID"].to_list()
        )
        logger.debug(
            f"Time anomalies detected: {len(time_anomalies)} transactions."
        )

        # 4. Invalid Format Checks
        format_anomalies = df.filter((pl.col("Quantity") <= 0))
        anomalies["format_anomalies"] = format_anomalies
        anomaly_transaction_ids.update(
            format_anomalies["Transaction ID"].to_list()
        )
        logger.debug(
            f"Format anomalies detected: {len(format_anomalies)} transactions."
        )

        # Filter out anomaly transactions from the clean DataFrame
        clean_df = clean_df.filter(
            ~pl.col("Transaction ID").is_in(list(anomaly_transaction_ids))
        )
        logger.info(
            f"Clean data size after anomaly removal: {clean_df.shape[0]} rows."
        )

    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise

    return anomalies, clean_df


def aggregate_daily_products(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates transaction data to a daily level (per product),
    summing the Quantity and grouping by (Date, Product Name).
    """
    df = df.with_columns(pl.col("Date").dt.date().alias("Date"))
    return df.group_by(["Date", "Product Name"]).agg(
        pl.col("Quantity").sum().alias("Total Quantity")
    )


def remove_duplicate_records(df: pl.DataFrame) -> pl.DataFrame:
    """Removes exact-duplicate Transaction IDs."""
    try:
        logger.info("Removing duplicate records...")
        original_count = df.height
        df = df.unique(subset=["Transaction ID"], maintain_order=True)
        removed_count = original_count - df.height
        if removed_count:
            logger.info(f"{removed_count} duplicate record(s) removed.")
        else:
            logger.info("No duplicate records found.")
        return df
    except Exception as e:
        logger.error(f"Error while removing duplicate records: {e}")
        raise


def extracting_time_series_and_lagged_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    For each row, computes additional time-series features:
      - day_of_week, is_weekend, etc.
      - lag_1, lag_7, rolling_mean_7 of 'Total Quantity'
    """
    try:
        if df.is_empty():
            logger.warning(
                "Input DataFrame is empty, returning an empty schema."
            )
            return pl.DataFrame(
                schema={
                    "Date": pl.Date,
                    "Product Name": pl.Utf8,
                    "Total Quantity": pl.Float64,
                    "day_of_week": pl.Int32,
                    "is_weekend": pl.Int8,
                    "day_of_month": pl.Int32,
                    "day_of_year": pl.Int32,
                    "month": pl.Int32,
                    "week_of_year": pl.Int32,
                    "lag_1": pl.Float64,
                    "lag_7": pl.Float64,
                    "rolling_mean_7": pl.Float64,
                }
            )

        # Ensure Date column is datetime type for feature extraction
        if "Date" in df.columns:
            df = df.with_columns(pl.col("Date").cast(pl.Datetime))

            df = df.with_columns(
                pl.col("Date").dt.weekday().alias("day_of_week"),
                (pl.col("Date").dt.weekday() > 5)
                .cast(pl.Int8)
                .alias("is_weekend"),
                pl.col("Date").dt.day().alias("day_of_month"),
                pl.col("Date").dt.ordinal_day().alias("day_of_year"),
                pl.col("Date").dt.month().alias("month"),
                pl.col("Date").dt.week().alias("week_of_year"),
            )
        else:
            logger.warning(
                "Date column not found, skipping datetime feature extraction"
            )
            return df
    except Exception as e:
        logger.error(
            f"Error extracting datetime features during feature engineering: {e}"
        )
        raise e

    try:
        # Only proceed with time series features if we have Total Quantity
        if "Total Quantity" in df.columns:
            # Sort by (Product Name, Date) for coherent time series ordering
            df = df.sort(["Product Name", "Date"]).with_columns(
                [
                    pl.col("Total Quantity")
                    .shift(1)
                    .over("Product Name")
                    .alias("lag_1"),
                    pl.col("Total Quantity")
                    .shift(7)
                    .over("Product Name")
                    .alias("lag_7"),
                    pl.col("Total Quantity")
                    .rolling_mean(window_size=7)
                    .over("Product Name")
                    .alias("rolling_mean_7"),
                ]
            )
        else:
            logger.warning(
                "Total Quantity column not found, skipping lagged features"
            )
            raise KeyError
    except Exception as e:
        logger.error(
            f"Error calculating lagged features during feature engineering: {e}"
        )
        raise e

    return df


def process_file(
    source_bucket_name: str,
    blob_name: str,
    destination_bucket_name: str,
    cache_bucket_name: str = None,
    delete_after_processing: bool = True,
) -> None:
    """
    Processes a single file through the entire data cleaning pipeline and uploads
    the result to GCS. Optionally deletes the source file after processing.

    Parameters:
        source_bucket_name (str): GCS bucket containing raw data.
        blob_name (str): Name of the blob/file to process.
        destination_bucket_name (str): Primary GCS bucket to store processed data.
        cache_bucket_name (str, optional): Secondary GCS bucket to cache processed data. If None, no caching is done.
        delete_after_processing (bool): Whether to delete source files after processing.
    """
    try:
        logger.info(f"Loading data from GCS: {blob_name}")
        df = load_bucket_data(source_bucket_name, blob_name)

        # Check if DataFrame is empty after loading
        if df.is_empty():
            logger.warning(
                f"File {blob_name} contains no data. Skipping processing."
            )
            return

        logger.info("Filling missing dates...")
        df = filling_missing_dates(df)

        logger.info("Converting feature types...")
        df = convert_feature_types(df)

        logger.info("Converting string columns to lowercase...")
        df = convert_string_columns_to_lowercase(df)

        logger.info("Standardizing Product Names...")
        df = standardize_product_name(df)

        logger.info("Filtering invalid product names...")
        df = filter_invalid_products(df, REFERENCE_PRODUCT_NAMES)

        # Check if DataFrame became empty after filtering
        if df.is_empty():
            logger.warning(
                "DataFrame became empty after filtering. Skipping further processing."
            )
            return

        if "Unit Price" in df.columns:
            logger.info("Filling missing Unit Prices...")
            df = filling_missing_cost_price(df)
        else:
            logger.info("Skipping filling missing Unit Price...")

        logger.info("Removing invalid records...")
        df = remove_invalid_records(df)

        logger.info("Removing Duplicate Records...")
        df = remove_duplicate_records(df)

        logger.info("Detecting Anomalies...")
        anomalies, df = detect_anomalies(df)

        # Check if any anomalies were found
        if any(not anom.is_empty() for anom in anomalies.values()):
            logger.info("Sending an Email for Alert...")
            try:
                send_anomaly_alert(anomalies=anomalies)
                logger.info("Anomaly alert sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send anomaly alert: {e}")
        else:
            logger.info("No anomalies detected. No email sent.")

        # Check if DataFrame became empty after anomaly detection
        if df.is_empty():
            logger.warning(
                "DataFrame became empty after anomaly detection. Skipping further processing."
            )
            return

        logger.info("Aggregating dataset to daily level...")
        df = aggregate_daily_products(df)

        logger.info("Performing Feature Engineering on Aggregated Data...")
        df = extracting_time_series_and_lagged_features(df)

        # Generate a consistent name for the output file (without timestamp)
        # This ensures we overwrite the existing file instead of creating a new one
        base_name = os.path.splitext(os.path.basename(blob_name))[0]
        dest_name = f"processed_{base_name}.csv"

        logger.info(
            f"Uploading cleaned data to GCS → {dest_name} (will overwrite if exists)"
        )
        upload_to_gcs(df, destination_bucket_name, dest_name)
        logger.info(
            f"Data cleaning completed! Cleaned data saved to GCS bucket: {destination_bucket_name}, "
            f"blob: {dest_name}"
        )

        # If cache bucket is provided, also upload to the cache bucket
        if cache_bucket_name:
            logger.info(
                f"Also uploading cleaned data to cache bucket: {cache_bucket_name}"
            )
            upload_to_gcs(df, cache_bucket_name, dest_name)
            logger.info(
                f"Data cached in bucket: {cache_bucket_name}, blob: {dest_name}"
            )

        if delete_after_processing:
            logger.info(
                f"Deleting source file {blob_name} from bucket {source_bucket_name}"
            )
            delete_success = delete_blob_from_bucket(
                source_bucket_name, blob_name
            )
            if delete_success:
                logger.info(f"Successfully deleted source file: {blob_name}")
            else:
                logger.warning(f"Failed to delete source file: {blob_name}")

        logger.info("Performing Post Validation...")
        # Also use a consistent name for statistics files
        stats_blob_name = f"stats_{base_name}.json"
        validation_passed = post_validation(df, stats_blob_name)

        if validation_passed:
            logger.info("Post-validation passed successfully.")
        else:
            logger.warning(
                "Post-validation detected issues with the processed data."
            )

    except Exception as e:
        logger.error(f"Processing file failed: {e}")
        logger.error(f"Failed file: {blob_name}")
        # Continue to the next file rather than raising
        return


def main(
    source_bucket_name: str = "full-raw-data",
    destination_bucket_name: str = "fully-processed-data",
    cache_bucket_name: str = None,
    delete_after_processing: bool = True,
) -> None:
    """
    Processes *all* files in the source GCS bucket, cleans them, and uploads them
    to the destination GCS bucket. Optionally deletes the source files after processing.

    Parameters:
        source_bucket_name (str): GCS bucket containing raw data.
        destination_bucket_name (str): GCS bucket to store processed data.
        cache_bucket_name (str, optional): Secondary GCS bucket to cache processed data. If None, no caching is done.
        delete_after_processing (bool): Whether to delete source files after processing.
    """
    try:
        blob_names = list_bucket_blobs(source_bucket_name)
        if not blob_names:
            logger.warning(f"No files found in bucket '{source_bucket_name}'")
            return

        # Process all files in the source bucket
        for blob_name in blob_names:
            logger.info(f"Processing file: {blob_name}")
            process_file(
                source_bucket_name=source_bucket_name,
                blob_name=blob_name,
                destination_bucket_name=destination_bucket_name,
                cache_bucket_name=cache_bucket_name,
                delete_after_processing=delete_after_processing,
            )

    except Exception as e:
        logger.error(f"Processing failed due to: {e}")
        raise e


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run preprocessing on data")
    parser.add_argument(
        "--source_bucket",
        type=str,
        default="full-raw-data",
        help="GCP bucket name containing source files",
    )
    parser.add_argument(
        "--destination_bucket",
        type=str,
        default="fully-processed-data",
        help="GCP bucket name for processed files",
    )
    parser.add_argument(
        "--cache_bucket",
        type=str,
        default=None,
        help="GCP bucket name for caching processed data",
    )
    parser.add_argument(
        "--delete_after",
        action="store_true",
        default=True,
        help="Delete source files after processing",
    )
    args = parser.parse_args()

    # Run main function with provided parameters
    main(
        source_bucket_name=args.source_bucket,
        destination_bucket_name=args.destination_bucket,
        cache_bucket_name=args.cache_bucket,
        delete_after_processing=args.delete_after,
    )
