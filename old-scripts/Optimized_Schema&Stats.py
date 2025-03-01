import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.model_selection import train_test_split
import tensorflow_data_validation as tfdv
from logger import logger


# File paths
DATASET_FILE_PATH = "transactions_20190103_20241231.xlsx"
SCHEMA_FILE_PATH = "transactions_schema.json"
STATS_FILE_PATH = "transactions_stats.json"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

# Load dataset
def load_data(file_path):
    logger.info("Loading dataset...")
    return pd.read_excel(file_path)

# Generate schema
def generate_schema(data):
    logger.info("Generating schema...")
    schema = {"columns": []}
    for column_name, dtype in data.dtypes.items():
        column_info = {
            "name": column_name,
            "type": dtype.name,
            "required": not data[column_name].isnull().any()
        }
        if column_name == "Date":
            column_info["format"] = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
        elif column_name in ["Store Location", "Product Name"]:
            column_info["allowed_values"] = list(data[column_name].unique())
        schema["columns"].append(column_info)
    return schema

# Generate statistics
def generate_statistics(data):
    logger.info("Generating dataset statistics...")
    stats = {}
    for column in data.columns:
        stats[column] = {
            "mean": data[column].mean() if pd.api.types.is_numeric_dtype(data[column]) else None,
            "std_dev": data[column].std() if pd.api.types.is_numeric_dtype(data[column]) else None,
            "min": data[column].min(),
            "max": data[column].max(),
            "missing_values": data[column].isnull().sum(),
            "unique_values": len(data[column].unique())
        }
    return stats

# Save data to JSON
def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info("Data saved to %s", file_path)

# Split data
def split_data(df, test_size=0.2):
    logger.info("Splitting dataset...")
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    logger.info("Train/Test split complete: Train(%d), Test(%d)", len(train_df), len(test_df))
    return train_df, test_df

# Perform TFDV analysis
def analyze_data_with_tfdv(train_df, test_df):
    logger.info("Analyzing data with TFDV...")
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    test_stats = tfdv.generate_statistics_from_dataframe(test_df)
    schema = tfdv.infer_schema(train_stats)
    anomalies = tfdv.validate_statistics(test_stats, schema)
    return train_stats, test_stats, schema, anomalies

# Main function
def main():
    df = load_data(DATASET_FILE_PATH)
    schema = generate_schema(df)
    stats = generate_statistics(df)
    save_to_json(schema, SCHEMA_FILE_PATH)
    save_to_json(stats, STATS_FILE_PATH)
    train_df, test_df = split_data(df)
    train_stats, test_stats, schema, anomalies = analyze_data_with_tfdv(train_df, test_df)
    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
