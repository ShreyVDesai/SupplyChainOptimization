import os
import pandas as pd
import json
import logging
import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
from tensorflow_metadata.proto.v0 import schema_pb2

# Configure Logging
log_file_path = "preprocessing_log.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)

# File Paths
SCHEMA_FILE = "schema.json"
STATS_FILE = "stats.json"


def load_data(file_path):
    """Loads data from CSV or XLSX"""
    logger.info(f"Loading dataset: {file_path}")
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Only .csv and .xlsx are allowed.")


def generate_statistics(data):
    """Generates statistics using TFDV"""
    logger.info("Generating dataset statistics using TFDV...")
    return tfdv.generate_statistics_from_dataframe(data)


def infer_schema(stats):
    """Infers schema using TFDV"""
    logger.info("Inferring schema from statistics...")
    return tfdv.infer_schema(stats)


def check_schema_changes(new_schema):
    """Compares new schema with existing schema and detects changes"""
    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE, "r") as f:
            old_schema = json.load(f)
        if str(new_schema) == str(old_schema):
            logger.info("No schema changes detected.")
            return False
        else:
            logger.warning("Schema changes detected!")
            return True
    return True  # First time running, assume schema is new


def save_json(data, file_path):
    """Saves data to JSON"""
    with open(file_path, "w") as f:
        json.dump(str(data), f, indent=4)
    logger.info(f"Saved {file_path}")


def analyze_data(file_path):
    """Main function to process the uploaded dataset"""
    try:
        # Load data
        df = load_data(file_path)

        # Split dataset
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

        # Generate statistics
        train_stats = generate_statistics(train_df)
        test_stats = generate_statistics(test_df)

        # Infer schema
        schema = infer_schema(train_stats)
        tfdv.display_schema(schema)

        # Validate test data against schema
        anomalies = tfdv.validate_statistics(test_stats, schema)

        # Log schema changes
        schema_changed = check_schema_changes(schema)

        # Save outputs
        save_json(schema, SCHEMA_FILE)
        save_json(train_stats, STATS_FILE)

        logger.info(f"Schema and statistics saved: {SCHEMA_FILE}, {STATS_FILE}")

        # Return True if schema changed or anomalies found
        return schema_changed or bool(anomalies.anomaly_info)

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return False


# result = analyze_data("transactions_20190103_20241231.xlsx")
# print("Schema change or anomaly detected:", result)
