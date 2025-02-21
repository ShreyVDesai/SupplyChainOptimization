import polars as pl
import io
from google.cloud import storage
from dotenv import load_dotenv
from typing import Dict, Tuple
from sendMail import send_email

load_dotenv()

def load_bucket_data(bucket_name, file_name):
    bucket = storage.Client().get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob_content = blob.download_as_string()
    return pl.read_excel(io.BytesIO(blob_content))

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
        "Unit Price": pl.Float64,
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
    Computes the most frequent unit price for each product at different time granularities.
    
    Parameters:
        df (pl.DataFrame): Input dataframe
        time_granularity (list): List of time-based features for grouping (e.g., ["Year", "Month", "Week"])
    
    Returns:
        pl.DataFrame: Mapping of (time + product) → most frequent unit price
    """

    return (
        df.drop_nulls(["Unit Price"])
        .group_by(time_granularity + ["Product Name"])
        .agg(pl.col("Unit Price").mode().first().alias("Most_Frequent_Cost"))
    )


def filling_missing_cost_price(df):
    """
    Fills missing 'Unit Price' values based on the most frequent price at different time granularities.

    Parameters:
        df (pl.DataFrame): Input dataframe

    Returns:
        pl.DataFrame: Updated dataframe with missing unit prices filled.
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


def upload_df_to_gcs(df, bucket_name, destination_blob_name):
    """
    Uploads a Polars DataFrame as a CSV file to a specified Google Cloud Storage bucket.
    
    :param bucket_name: Name of the GCS bucket
    :param df: The Polars DataFrame to upload
    :param destination_blob_name: Destination path in GCS where the file will be stored
    """

    bucket = storage.Client().get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    csv_data = df.write_csv()
    blob.upload_from_string(csv_data, content_type='text/csv')

def calculate_zscore(series: pl.Series) -> pl.Series:
    """Calculate Z-score for a series"""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pl.Series([0] * len(series))
    return (series - mean) / std
    
def iqr_bounds(series: pl.Series) -> Tuple[float, float]:
    """Calculate IQR bounds for a series"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3.0 * iqr
    upper_bound = q3 + 3.0 * iqr
    return max(0, lower_bound), upper_bound

def detect_anomalies(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Detect anomalies in transaction data using both IQR and Z-score methods.
    
    Parameters:
    df (pl.DataFrame): DataFrame containing transaction data with columns:
        - Date
        - Unit Price
        - Transaction ID
        - Quantity
        - Producer ID
        - Store Location
        - Product Name
    
    Returns:
    Dict[str, pl.DataFrame]: Dictionary containing different types of anomalies detected
    """
    
    anomalies = {}
    clean_df = df.clone()
    anomaly_transaction_ids = set() 
    
    # 1. Missing Values
    missing_counts = []
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            missing_counts.append({"column": col, "null_count": null_count})
            null_rows = df.filter(pl.col(col).is_null())
            anomaly_transaction_ids.update(null_rows['Transaction ID'].to_list())
    
    anomalies['missing_values'] = pl.DataFrame(missing_counts)
    
    # 3. Price Anomalies (by Product)
    price_anomalies = []
    for product in df['Product Name'].unique():
        product_data = df.filter(pl.col('Product Name') == product)
        
        # Z-score method
        # zscore_prices = calculate_zscore(product_data['Unit Price'])
        # zscore_anomalies = product_data.filter(
        #     (zscore_prices.abs() > 3)
        # )
        
        # IQR method
        lower_bound, upper_bound = iqr_bounds(product_data['Unit Price'])
        iqr_anomalies = product_data.filter(
            (pl.col('Unit Price') < lower_bound) | 
            (pl.col('Unit Price') > upper_bound)
        )
        
        # Combine both methods
        # combined_anomalies = pl.concat([zscore_anomalies, iqr_anomalies]).unique()
        # if len(combined_anomalies) > 0:
        #     price_anomalies.append(combined_anomalies)
        price_anomalies.append(iqr_anomalies)
        anomaly_transaction_ids.update(iqr_anomalies['Transaction ID'].to_list())
    
    if price_anomalies:
        anomalies['price_anomalies'] = pl.concat(price_anomalies)
    else:
        anomalies['price_anomalies'] = pl.DataFrame()
    
    # 4. Quantity Anomalies (by Product)
    quantity_anomalies = []
    for product in df['Product Name'].unique():
        product_data = df.filter(pl.col('Product Name') == product)
        
        # Z-score method
        # zscore_quantities = calculate_zscore(product_data['Quantity'])
        # zscore_anomalies = product_data.filter(
        #     (zscore_quantities.abs() > 3)
        # )
        
        # IQR method
        lower_bound, upper_bound = iqr_bounds(product_data['Quantity'])

        iqr_anomalies = product_data.filter(
            (pl.col('Quantity') < lower_bound) | 
            (pl.col('Quantity') > upper_bound)
        )
        
        # Combine both methods
        # combined_anomalies = pl.concat([zscore_anomalies, iqr_anomalies]).unique()
        # if len(combined_anomalies) > 0:
        #     quantity_anomalies.append(combined_anomalies)
        quantity_anomalies.append(iqr_anomalies)
        anomaly_transaction_ids.update(iqr_anomalies['Transaction ID'].to_list())
    
    if quantity_anomalies:
        anomalies['quantity_anomalies'] = pl.concat(quantity_anomalies)
    else:
        anomalies['quantity_anomalies'] = pl.DataFrame()
    
    # 5. Time Pattern Anomalies
    df = df.with_columns([
        pl.col('Date').cast(pl.Datetime).alias('datetime'),
        pl.col('Date').cast(pl.Datetime).dt.hour().alias('hour')
    ])
    
    # Detect transactions outside normal business hours (assuming 6AM-10PM)
    time_anomalies = df.filter(
        (pl.col('hour') < 6) | (pl.col('hour') > 22)
    )
    anomalies['time_anomalies'] = time_anomalies
    anomaly_transaction_ids.update(time_anomalies['Transaction ID'].to_list())
    
    # 6. Invalid Format Checks
    format_anomalies = df.filter(
        # Check for negative prices
        (pl.col('Unit Price') <= 0) |
        # Check for negative quantities
        (pl.col('Quantity') <= 0)
    )
    anomalies['format_anomalies'] = format_anomalies
    anomaly_transaction_ids.update(format_anomalies['Transaction ID'].to_list())
    
    clean_df = clean_df.filter(~pl.col('Transaction ID').is_in(list(anomaly_transaction_ids)))
    
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


def main(input, output_file, cloud=True):
    """Executes all cleaning steps."""
    bucket_name = 'mlops-data-storage-000' 
    source_blob_name = 'generated_training_data/transactions_20190103_20241231.xlsx'
    destination_blob_name = 'cleaned_data/cleanedData.csv'
    if cloud:
        df = load_bucket_data(bucket_name, source_blob_name)
    else:
        df = load_data(input)

    df = clean_dates(df)

    df = convert_feature_types(df)

    df = fill_missing_value_with_unknown(df, ["Producer ID", "Store Location", "Transaction ID"])

    # df = replace_digits_in_string_columns(df)

    df = convert_string_columns_to_lowercase(df)

    df = filling_missing_cost_price(df)

    df = remove_invalid_records(df)

    anomalies, df = detect_anomalies(df)
    
    # print("Cleaning Store Location Data...")
    # df = clean_store_location(df)
    
    # print("Cleaning Product Name Data...")
    # df = clean_product_names(df)
    
    # print("Filtering Valid Records...")
    # df = filter_valid_records(df, ["Quantity", "Unit Price"])
    
    # print("Cleaning Date Data...")
    # df = clean_dates(df)

    message_parts = ["Hi, we removed the following data from your excel:"]
    for anomaly_type, anomaly_df in anomalies.items():
        if not anomaly_df.is_empty():
            message_parts.append(f"\n{anomaly_type}:")
            message_parts.append(anomaly_df)
    
    message_parts.append("Thank you!")
    # send_email("anusheelvs5050@gmail.com", message_parts, subject="Anomaly Data")

    df = df.with_columns(pl.col("Date").dt.date().alias("Date"))
    df = aggregate_daily_products(df)


    
    # print("Saving Cleaned Data...")
    # if cloud:
    #     upload_df_to_gcs(bucket_name, df, destination_blob_name)
    # else:
    #     save_cleaned_data(df, output_file)

    # print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    input_file = "messy_transactions_20190103_20241231.xlsx"  # Change to actual file
    output_file = "../../data/cleaned_data.csv"
    main(input_file, output_file, True)
