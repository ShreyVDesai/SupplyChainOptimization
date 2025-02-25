import numpy as np
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz
import polars as pl
import logging
import io
from google.cloud import storage
from dotenv import load_dotenv
from typing import Dict, Tuple
from sendMail import send_email

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Reference product names (correct names from the dataset)
REFERENCE_PRODUCT_NAMES = [
    "milk", "coffee", "wheat", "chocolate", "beef", "sugar", "corn", "soybeans"
]


def load_bucket_data(bucket_name: str, file_name: str) -> pl.DataFrame:
    """
    Loads data from a specified file in a Google Cloud Storage bucket and returns it as a Polars DataFrame.
    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        file_name (str): The name of the file within the bucket, including extension.

    Returns:
        pl.DataFrame: The content of the Excel file as a Polars DataFrame.

    Raises:
        Exception: If an error occurs while accessing the bucket or reading the file.
    """
    try:
        bucket = storage.Client().get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob_content = blob.download_as_string()
        data_frame = pl.read_excel(io.BytesIO(blob_content))
        logging.info(f"'{file_name}' from bucket '{bucket_name}' successfully read into DataFrame.")

        return data_frame

    except Exception as e:
        logging.error(f"Error occurred while loading data from bucket '{bucket_name}', file '{file_name}': {e}")
        raise


def load_data(file_path: str) -> pl.DataFrame:
    """
    Loads the dataset from the given file path.
    
    Parameters:
        file_path (str): Path to the input file.

    Returns:
        pl.DataFrame: Loaded DataFrame.
    """
    try:
        if file_path.lower().endswith(".xlsx"):
            df = pl.read_excel(file_path)
        
        logging.info(f"Data successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    
    except FileNotFoundError:
        logging.error(f"File Not Found: {file_path}")

    except Exception as e:
        logging.error(f"Fail to load data due to: {e}")
        raise e

        


def convert_feature_types(df: pl.DataFrame) -> pl.DataFrame:
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
        "Unit Price": pl.Float64,
        "Quantity": pl.Int64,
        "Transaction ID": pl.Utf8,
        "Store Location": pl.Utf8,
        "Product Name": pl.Utf8,
        "Producer ID": pl.Utf8
    }

    try:
        # Ensure required columns exist
        missing_columns = [col for col in expected_dtypes if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in DataFrame: {missing_columns}")
            
        # Convert columns to expected data types
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(dtype))

        logging.info("Feature types converted successfully.")
        return df
    except Exception as e:
        logging.error(f"Unexpected error during feature type conversion.", exc_info=True)
        raise


def fill_missing_value_with_unknown(df: pl.DataFrame, features: list) -> pl.DataFrame:
    """Fills missing values in specified columns with the string 'Unknown'.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
        features (list): List of feature names to process

    Returns:
        pl.DataFrame: Updated DataFrame with missing values replaced
    """
    try:
        logging.info(f"Filling missing values with unknown for given features: {features}")

        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            raise KeyError(f"Features missing from the dataframe: {missing_features}")

        for feature in features:
            try:
                df = df.with_columns(
                    pl.col(feature)
                    .cast(pl.Utf8)
                    .fill_null("Unknown")
                )
                

            except Exception as e:
                logging.error(f"Error filling missing values in: {feature}")
        
        logging.info("Sucessfully filled missing values with 'unknown'")
        return df
    
    except Exception as e:
        logging.error("Error filling missing values with unknown.")
        raise e



def convert_string_columns_to_lowercase(df: pl.DataFrame) -> pl.DataFrame:
    """Converts all string-type columns to lowercase for consistency.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Updated DataFrame with lowercase strings
    """
    try:
        # Detect all string-type columns dynamically
        logging.info("Converting all string-type columns to lowercase...")
        string_features = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]

        if not string_features:
            logging.warning("No string columns detected. Skipping conversion.")
            return df

        df = df.with_columns(
            [pl.col(feature).str.to_lowercase() for feature in string_features]
        )

        return df
    except Exception as e:
        logging.error(f"Unexpected error during string conversion: {e}")
        raise e



def standardize_date_format(df: pl.DataFrame, date_column: str = "Date") -> pl.DataFrame:
    """
    Standardizes date formats in the given column to a consistent format: '%Y-%m-%d'.
    - Converts multiple formats into a single standard format.
    - Handles missing or incorrect values gracefully.
    - Ensures uniform datetime format.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame.
        date_column (str): Column name containing date values.
    
    Returns:
        pl.DataFrame: Updated DataFrame with standardized date formats.
    """
    try:
        logging.info("Standardizing date formats...")
        
        if date_column not in df.columns:
            logging.error(f"Column '{date_column}' not found in DataFrame.")
            return df
        
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

        # Ensure the column is cast to Datetime while handling null values
        df = df.with_columns(
            pl.when(pl.col(date_column).is_not_null())
            .then(pl.col(date_column).cast(pl.Datetime))
            .otherwise(None)
            .alias(date_column)
        )
        
        # Log any null values after standardization
        null_count = df[date_column].null_count()
        if null_count > 0:
            logging.warning(f"Date column '{date_column}' has {null_count} null values after conversion. Check data format.")
        
        logging.info("Date standardization completed successfully.")
        return df
    
    except Exception as e:
        logging.error(f"Unexpected error during date processing: {e}")
        raise e



def detect_date_order(df: pl.DataFrame, date_column: str = "Date") -> str:
    """
    Identifies the ordering of the 'Date' column, even if some dates are missing.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame.
        date_column (str): Column name containing date values.

    Returns:
        str: 
            - "Ascending" if dates are mostly in increasing order.
            - "Descending" if dates are mostly in decreasing order.
            - "Random" if no clear pattern is detected.
    """
    try:
        if date_column not in df.columns:
            logging.error(f"Column '{date_column}' not found in DataFrame.")
            return "Unknown"
        
        # Ensure Date column is in Datetime format
        df = df.with_columns(pl.col(date_column).cast(pl.Datetime))

        # Extract the non-null date series
        date_series = df.drop_nulls(date_column)[date_column].to_list()
        
        # Compute the number of ascending and descending transitions efficiently
        asc_count = sum(1 for i in range(len(date_series) - 1) if date_series[i] <= date_series[i + 1])
        desc_count = sum(1 for i in range(len(date_series) - 1) if date_series[i] >= date_series[i + 1])

        # Determine the dominant ordering
        if asc_count > desc_count:
            return "Ascending"
        elif desc_count > asc_count:
            return "Descending"
        else:
            return "Random"
    
    except Exception as e:
        logging.error(f"Error detecting date order: {e}")
        raise e

        

def filling_missing_dates(df: pl.DataFrame, date_column: str = "Date") -> pl.DataFrame:
    """
    Fills missing dates based on detected order or a specified strategy.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame.
        date_column (str): Column name containing date values.
    
    Returns:
        pl.DataFrame: Updated DataFrame with missing dates filled or handled.
    """
    try:
        # Ensure Date column is in correct format before filling
        df = standardize_date_format(df, date_column)
        
        order_type = detect_date_order(df, date_column)
        
        if df[date_column].null_count() > 0:
            logging.warning(f"{df[date_column].null_count()} missing date values before filling.")

        if order_type == "Ascending":
            logging.info("Ascending Order Detected: Using Forward-Fill")
            # df = df.with_columns(pl.col(date_column).fill_null(strategy="forward"))
            df = df.with_columns(
                pl.col(date_column).interpolate()
            )
        elif order_type == "Descending":
            logging.info("Descending Order Detected: Using Backward-Fill")
            # df = df.with_columns(pl.col(date_column).fill_null(strategy="backward"))
            df = df.with_columns(
                pl.col(date_column).reverse().interpolate().reverse()
            )
        else:
            logging.warning("Random Order Detected: Dropping Missing Dates")
            df = df.filter(pl.col(date_column).is_not_null())
        
        if df[date_column].null_count() > 0:
            logging.warning(f"{df[date_column].null_count()} missing date values remain after filling.")
        
        return df
    
    except Exception as e:
        logging.error(f"Error filling missing dates: {e}")
        raise e




def remove_future_dates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Removes records where the date is in the future compared to today's date.

    Parameters:
        df (pl.DataFrame): Input DataFrame
        date_column (str): Name of the date column (default is "Date")

    Returns:
        pl.DataFrame: Filtered DataFrame with future dates removed.
    """
    try:
        today = datetime.today().date()

        # Keeping only today's data
        return df.filter(
            pl.col("Date").dt.date() <= today 
        )
    except Exception as e:
        logging.error(f"Unexpected error during removing future dates: {e}")
        raise e



def extract_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract Year, Month and Week of year from date column.
    """
    try:
        return df.with_columns(
            pl.col("Date").dt.year().alias("Year"),
            pl.col("Date").dt.month().alias("Month"),
            pl.col("Date").dt.week().alias("Week_of_year")
        )
    except Exception as e:
        logging.error(f"Error extracting datetime feature.")
        raise e


def compute_most_frequent_price(df: pl.DataFrame, time_granularity: list) -> pl.DataFrame:
    """
    Computes the most frequent unit price for each product at different time granularities.
    
    Parameters:
        df (pl.DataFrame): Input dataframe
        time_granularity (list): List of time-based features for grouping (e.g., ["Year", "Month", "Week"])
    
    Returns:
        pl.DataFrame: Mapping of (time + product) → most frequent unit price
    """
    try:
        return (
            df.drop_nulls(["Unit Price"])
            .group_by(time_granularity + ["Product Name"])
            .agg(pl.col("Unit Price").mode().first().alias("Most_Frequent_Cost"))
        )
    except Exception as e:
        logging.error(f"Error computing most frequent price: {e}")



def filling_missing_cost_price(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fills missing 'Unit Price' values based on the most frequent price at different time granularities.

    Parameters:
        df (pl.DataFrame): Input dataframe

    Returns:
        pl.DataFrame: Updated dataframe with missing unit prices filled.
    """
    
    try:
        logging.info("Filling missing Unit Price values...")

        # Extract time features dynamically
        df = extract_datetime_features(df)

        # Compute most frequent prices at different time levels
        price_by_week = compute_most_frequent_price(df, ["Year", "Month", "Week_of_year"])
        price_by_month = compute_most_frequent_price(df, ["Year", "Month"])
        price_by_year = compute_most_frequent_price(df, ["Year"])
        

        # Merge with original dataframe to fill missing values
        df = df.join(price_by_week, on=["Year", "Month", "Week_of_year", "Product Name"], how="left")
        df = df.with_columns(
            pl.when(pl.col("Unit Price").is_null())
            .then(pl.col("Most_Frequent_Cost"))
            .otherwise(pl.col("Unit Price"))
            .alias("Unit Price")
        ).drop("Most_Frequent_Cost")


        df = df.join(price_by_month, on=["Year", "Month", "Product Name"], how="left")
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


        # If still null, set to "Unknown" or a default value (e.g., 0)
        df = df.with_columns(
            pl.col("Unit Price").fill_null(0)
        )
        
        logging.info("Unit Price filling completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Unexpected error while filling missing Unit Prices: {e}")
        raise e



# If Quantity or Product Name is missing, remove those records.
def remove_invalid_records(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove records if Quantity or Product Name has invalid input.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Filtered DataFrame with invalid records removed.
    """
    try:
        return df.filter(
            # (pl.col("Quantity") > 0) &
            pl.col("Quantity").is_not_null() &
            pl.col("Product Name").is_not_null()
        )
    except Exception as e:
        logging.error(f"Error handling remove_invalid_records function: {e}")
        raise e



def standardize_product_name(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardizes product names by:
    - Converting to lowercase
    - Stripping extra spaces
    - Replacing '@' with 'a' and '0' with 'o'
    - Fixing known typos
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Updated DataFrame with standardized product names.
    """
    try:
        logging.info("Standardizing product name.")
        df = df.with_columns(
            pl.col("Product Name")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_lowercase()
            .str.replace_all("@", "a")
            .str.replace_all("0", "o")
            .alias("Product Name")
        )

        return df
    except Exception as e:
        logging.error(f"Error standardizing product name: {e}")
        raise e



def apply_fuzzy_correction(df: pl.DataFrame, reference_list: list, threshold: int = 80) -> pl.DataFrame:
    """
    Uses fuzzy matching to correct near-duplicate product names.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
        reference_list (list): List of correct product names
        threshold (int): Similarity threshold (0-100, higher means stricter matching)
    
    Returns:
        pl.DataFrame: DataFrame with corrected product names.
    """
    try:
        product_names = df["Product Name"].unique().to_list()

        name_mapping = {}

        for name in product_names:
            match_result = process.extractOne(name, reference_list, scorer=fuzz.ratio)
            
            if match_result and len(match_result) > 1:
                match, score = match_result[:2]
                if score >= threshold:
                    name_mapping[name] = match

        df = df.with_columns(
            pl.col("Product Name").replace(name_mapping).alias("Product Name")
        )
        logging.info(f"Fuzzy matching completed. {len(name_mapping)} product names corrected.")
        return df
    
    except Exception as e:
        logging.error(f"Unexpected error during fuzzy matching: {e}")
        raise e


def clean_and_correct_product_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full pipeline to standardize and correct product names.
    
    Steps:
    - Standardize case and fix typos.
    - Apply fuzzy matching for near-duplicate correction.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame
    
    Returns:
        pl.DataFrame: Cleaned and corrected DataFrame.
    """
    df = standardize_product_name(df)
    df = apply_fuzzy_correction(df, REFERENCE_PRODUCT_NAMES)
    
    return df




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





def upload_df_to_gcs(df: pl.DataFrame, bucket_name: str, destination_blob_name: str) -> None:
    """
    Uploads a DataFrame to Google Cloud Storage (GCS) as a CSV file.
    Args:
        df (polars.DataFrame): The DataFrame to upload.
        bucket_name (str): The name of the GCS bucket where the file should be stored.
        destination_blob_name (str): The desired name for the file in GCS.

    Returns:
        None

    Raises:
        google.cloud.exceptions.GoogleCloudError: If the upload fails.
        Exception: If any other error occurs during the process.
    """
    try:
        logging.info("Starting upload to GCS. Bucket: %s, Blob: %s", bucket_name, destination_blob_name)
        bucket = storage.Client().get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        csv_data = df.write_csv()
        blob.upload_from_string(csv_data, content_type='text/csv')
        
        logging.info("Upload successful to GCS. Blob name: %s", destination_blob_name)
    
    except Exception as e:
        logging.error("Error uploading DataFrame to GCS. Error: %s", e)
        raise

def calculate_zscore(series: pl.Series) -> pl.Series:
    """Calculate Z-score for a series"""
    try:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pl.Series([0] * len(series))
        
        return (series - mean) / std

    except Exception as e:
        logging.error(f"Error calculating Z-score: {e}")
        raise


def iqr_bounds(series: pl.Series) -> Tuple[float, float]:
    """Calculate IQR bounds for a series"""
    try:
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (2.0 * iqr)

        # Log bounds
        logging.debug(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

        return max(0, lower_bound), upper_bound

    except Exception as e:
        logging.error(f"Error calculating IQR bounds: {e}")
        raise


def detect_anomalies(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Detect anomalies in transaction data per product per day using IQR.
    
    Parameters:
    df (pl.DataFrame): DataFrame containing transaction data.
    
    Returns:
    Dict[str, pl.DataFrame]: Dictionary containing different types of anomalies detected
    """
    
    anomalies = {}
    clean_df = df.clone()
    anomaly_transaction_ids = set()
    
    try:
        # Ensure Date column is in the correct format
        df = df.with_columns([
            pl.col('Date').cast(pl.Datetime).alias('datetime'),
            pl.col('Date').cast(pl.Datetime).dt.date().alias('date_only'),
            pl.col('Date').cast(pl.Datetime).dt.hour().alias('hour')
        ])
        
        # 1. Price Anomalies (by Product and Date)
        price_anomalies = []
        # Get unique combinations of product and date
        product_date_combinations = df.select(['Product Name', 'date_only']).unique()
        
        for row in product_date_combinations.iter_rows(named=True):
            product = row['Product Name']
            date = row['date_only']
            
            # Filter data for specific product and date
            product_date_data = df.filter(
                (pl.col('Product Name') == product) & 
                (pl.col('date_only') == date)
            )
            
            # Only proceed if we have enough data points for this product/date combination
            if len(product_date_data) >= 4:  # Minimum sample size for meaningful IQR
                # IQR method
                lower_bound, upper_bound = iqr_bounds(product_date_data['Unit Price'])
                iqr_anomalies = product_date_data.filter(
                    (pl.col('Unit Price') < lower_bound) | 
                    (pl.col('Unit Price') > upper_bound)
                )
                
                if len(iqr_anomalies) > 0:
                    price_anomalies.append(iqr_anomalies)
                    anomaly_transaction_ids.update(iqr_anomalies['Transaction ID'].to_list())
        
        if price_anomalies:
            anomalies['price_anomalies'] = pl.concat(price_anomalies)
        else:
            anomalies['price_anomalies'] = pl.DataFrame()
        logging.debug(f"Price anomalies detected: {len(price_anomalies)} products.")
        
        # 2. Quantity Anomalies (by Product and Date)
        quantity_anomalies = []
        
        for row in product_date_combinations.iter_rows(named=True):
            product = row['Product Name']
            date = row['date_only']
            
            # Filter data for specific product and date
            product_date_data = df.filter(
                (pl.col('Product Name') == product) & 
                (pl.col('date_only') == date)
            )
            
            # Only proceed if we have enough data points for this product/date combination
            if len(product_date_data) >= 4:  # Minimum sample size for meaningful IQR
                # IQR method
                lower_bound, upper_bound = iqr_bounds(product_date_data['Quantity'])
                iqr_anomalies = product_date_data.filter(
                    (pl.col('Quantity') < lower_bound) | 
                    (pl.col('Quantity') > upper_bound)
                )
                
                if len(iqr_anomalies) > 0:
                    quantity_anomalies.append(iqr_anomalies)
                    anomaly_transaction_ids.update(iqr_anomalies['Transaction ID'].to_list())
        
        if quantity_anomalies:
            anomalies['quantity_anomalies'] = pl.concat(quantity_anomalies)
        else:
            anomalies['quantity_anomalies'] = pl.DataFrame()
        logging.debug(f"Quantity anomalies detected: {len(quantity_anomalies)} products.")

        # 3. Time Pattern Anomalies
        time_anomalies = df.filter(
            (pl.col('hour') < 6) | (pl.col('hour') > 22)
        )
        anomalies['time_anomalies'] = time_anomalies
        anomaly_transaction_ids.update(time_anomalies['Transaction ID'].to_list())
        logging.debug(f"Time anomalies detected: {len(time_anomalies)} transactions.")
        
        # 4. Invalid Format Checks
        format_anomalies = df.filter(
            (pl.col('Unit Price') <= 0) |
            (pl.col('Quantity') <= 0)
        )
        anomalies['format_anomalies'] = format_anomalies
        anomaly_transaction_ids.update(format_anomalies['Transaction ID'].to_list())
        
        logging.debug(f"Format anomalies detected: {len(format_anomalies)} transactions.")

        clean_df = clean_df.filter(~pl.col('Transaction ID').is_in(list(anomaly_transaction_ids)))
        logging.info(f"Clean data size after anomaly removal: {clean_df.shape[0]} rows.")
    
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        raise
    
    return anomalies, clean_df


    


def aggregate_daily_products(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates transaction data to show total quantity per product per day.
    
    Parameters:
    df (pl.DataFrame): Input DataFrame with columns Date, Unit Price, Quantity, and Product Name
    
    Returns:
    pl.DataFrame: Aggregated DataFrame with daily totals per product
    """
    return (
        df.group_by(["Date", "Product Name", "Unit Price"])
        .agg(
            pl.col("Quantity").sum().alias("Total Quantity")
        )
        .sort(["Date", "Product Name"])
    )

def remove_duplicate_records(df: pl.DataFrame) -> pl.DataFrame:
    """
    Removes duplicate transaction records if exists.

    Parameters:
        df (pl.DataFrame): Input DataFrame
    Returns:
        pl.DataFrame: DataFrame without duplicate records.
    """
    try:
        logging.info("Removing duplicate records...")
        df = df.unique(subset=["Transaction ID"], maintain_order=True)
        logging.info("Duplicate records removed.")
        return df
    except Exception as e:
        logging.error(f"Error while removing duplicate records: {e}")
        raise e


def save_cleaned_data(df: pl.DataFrame, output_file: str) -> None:
    """Saves the cleaned data to a CSV file."""
    try:
        df.write_csv(output_file)
    except Exception as e:
        logging.error(f"Error saving processed DataFrame: {e}")
        raise e

def main(input_file: str = "../../transaction/transactions_20190103_20241231.xlsx", 
         output_file: str  = "../../data/cleaned_data.csv", 
         bucket_name: str = 'mlops-data-storage-000', 
         source_blob_name: str = 'generated_training_data/transactions_20190103_20241231.xlsx', 
         destination_blob_name: str = 'cleaned_data/cleanedData.csv', 
         cloud: bool = False) -> None:
    """
    Executes all data cleaning steps in sequence.

    Parameters:
        input_file (str): Path to the input dataset.
        output_file (str): Path to save the cleaned dataset.

    Raises:
        RuntimeError: If any step fails during execution.
    """

    try:
        logging.info("Loading data...")
        if cloud:
            df = load_bucket_data(bucket_name, source_blob_name)
        else:
            df = load_data(input_file)

        # logging.info("Standardizing date formats...")
        # df = standardize_date_format(df)

        logging.info("Filling missing dates...")
        df = filling_missing_dates(df)

        logging.info("Converting feature types...")
        df = convert_feature_types(df)

        logging.info("Filling missing values with 'Unknown'...")
        df = fill_missing_value_with_unknown(df, ["Producer ID", "Store Location", "Transaction ID"])

        logging.info("Converting string columns to lowercase...")
        df = convert_string_columns_to_lowercase(df)

        logging.info("Filling missing Unit Prices using most frequent price...")
        df = filling_missing_cost_price(df)

        logging.info("Removing invalid records...")
        df = remove_invalid_records(df)

        logging.info("Standardizing and correcting product names...")
        df = clean_and_correct_product_names(df)

        logging.info("Removing Duplicate Records...")
        df = remove_duplicate_records(df)

        anomalies, df = detect_anomalies(df)

        message_parts = ["Hi, we removed the following data from your excel:"]
        for anomaly_type, anomaly_df in anomalies.items():
            if not anomaly_df.is_empty():
                message_parts.append(f"\n{anomaly_type}:")
                message_parts.append(anomaly_df)
        
        message_parts.append("Thank you!")
        send_email("patelmit640@gmail.com", message_parts, subject="Anomaly Data")

        df = df.with_columns(pl.col("Date").dt.date().alias("Date"))
        df = aggregate_daily_products(df)
        
        logging.info("Saving cleaned data...")
        if cloud:
            upload_df_to_gcs(df, bucket_name, destination_blob_name)
        else:
            save_cleaned_data(df, output_file)
        logging.info(f"Data cleaning completed! Cleaned data saved to: {output_file}")

    except Exception as e:
        logging.error(f"Processing failed due to: {e}")
        raise e


if __name__ == "__main__":
    main()
