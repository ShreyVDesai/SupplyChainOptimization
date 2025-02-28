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
# Define logger
logger = logging.getLogger(__name__)

# File Paths - use script directory for storing schema files
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEMA_FILE = os.path.join(script_dir, "schema.json")
STATS_FILE = os.path.join(script_dir, "stats.json")


def load_data(file_path):
    """Loads data from CSV or XLSX"""
    logger.info(f"Loading dataset: {file_path}")
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        logger.warning(f"Attempting to load unknown format: {file_path}")
        # Try to infer format based on content
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        else:
            try:
                return pd.read_csv(file_path)
            except:
                try:
                    return pd.read_excel(file_path)
                except:
                    raise ValueError(
                        f"Failed to load file: {file_path}. Unsupported format."
                    )


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
        try:
            with open(SCHEMA_FILE, "r") as f:
                old_schema = json.load(f)
            if str(new_schema) == str(old_schema):
                logger.info("No schema changes detected.")
                return False
            else:
                logger.warning("Schema changes detected!")
                return True
        except Exception as e:
            logger.error(f"Error reading existing schema: {e}")
            return True  # Assume schema change if we can't read the old one
    return True  # First time running, assume schema is new


def save_json(data, file_path):
    """Saves data to JSON"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(str(data), f, indent=4)
        logger.info(f"Saved {file_path}")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")


def analyze_data(file_path):
    """Main function to process the uploaded dataset"""
    try:
        logger.info(f"Starting analysis of file: {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return True  # Return error

        # Load data
        df = load_data(file_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")

        # Check if there's enough data to split
        if len(df) < 5:  # Minimal threshold
            logger.warning("Dataset too small for proper validation")
            return True

        # Split dataset
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

        # Generate statistics
        train_stats = generate_statistics(train_df)
        test_stats = generate_statistics(test_df)

        # Infer schema
        schema = infer_schema(train_stats)

        # Validate test data against schema
        anomalies = tfdv.validate_statistics(test_stats, schema)

        # Log schema changes
        schema_changed = check_schema_changes(schema)

        # Count anomalies
        anomaly_count = (
            len(anomalies.anomaly_info) if hasattr(anomalies, "anomaly_info") else 0
        )
        logger.info(f"Found {anomaly_count} anomalies in data validation")

        # Save outputs
        save_json(schema, SCHEMA_FILE)
        save_json(train_stats, STATS_FILE)

        logger.info(f"Schema and statistics saved: {SCHEMA_FILE}, {STATS_FILE}")

        has_issues = schema_changed or anomaly_count > 0
        logger.info(f"Validation complete. Has issues: {has_issues}")

        # Return True if schema changed or anomalies found
        return has_issues

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return True  # Return error condition if exception occurs


if __name__ == "__main__":
    # For testing the script directly
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = analyze_data(file_path)
        print(f"Schema change or anomaly detected: {result}")
    else:
        print("Usage: python DataValidation_Schema&Stats.py <file_path>")
