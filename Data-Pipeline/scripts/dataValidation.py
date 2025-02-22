import polars as pl
from typing import List, Dict, Tuple


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
        (pl.col('Quantity') <= 0) |
        # Check for invalid Producer IDs
        (pl.col('Producer ID') <= 0)
    )
    anomalies['format_anomalies'] = format_anomalies
    anomaly_transaction_ids.update(format_anomalies['Transaction ID'].to_list())
    
    clean_df = clean_df.filter(~pl.col('Transaction ID').is_in(list(anomaly_transaction_ids)))
    
    return anomalies, clean_df

# Example usage:
def example_usage():    
    df = pl.read_excel("E:/MLOps/SupplyChainOptimization/transaction/transactions_20190103_20241231.xlsx")    
    anomalies, cleaned_df = detect_anomalies(df)
    # Print results
    for anomaly_type, anomaly_df in anomalies.items():
        print(f"\n{anomaly_type}:")
        print(anomaly_df)

if __name__ == "__main__":
    example_usage()

# import time
# start_time = time.time()
# df = pl.read_excel("E:/MLOps/SupplyChainOptimization/transaction/transactions_20190103_20241231.xlsx")
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# time.sleep(2)    