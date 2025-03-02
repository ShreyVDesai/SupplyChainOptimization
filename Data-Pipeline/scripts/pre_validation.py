import os
import polars as pl
import pandas as pd
import argparse
from logger import logger
from utils import send_email, load_bucket_data, load_data
from google.cloud import storage
from dotenv import load_dotenv
from utils import setup_gcp_credentials

load_dotenv()

# Pre-validation expected columns
PRE_VALIDATION_COLUMNS = [
    "Date",
    "Unit Price",
    "Transaction ID",
    "Quantity",
    "Producer ID",
    "Store Location",
    "Product Name",
]


def collect_validation_errors(df, missing_columns, error_indices, error_reasons):
    """
    Collect validation errors and update error indices and reasons.

    Parameters:
      df: The DataFrame being validated.
      missing_columns: List of columns that are missing.
      error_indices: A set to store indices of rows with errors.
      error_reasons: A dictionary to store error reasons for each row.
    """
    if missing_columns:
        # If columns are missing, mark all rows as having errors
        for idx in range(len(df)):
            error_indices.add(idx)
            error_reasons[idx] = [f"Missing columns: {', '.join(missing_columns)}"]


def validate_data(df):
    """
    Validate the DataFrame by checking if it contains all required columns.

    Returns:
      bool: True if all checks pass, False otherwise.
    """
    try:
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Check if all required columns are present
        missing_columns = [
            col for col in PRE_VALIDATION_COLUMNS if col not in df.columns
        ]

        if not missing_columns:
            logger.info("Data validation passed successfully.")
            return True
        else:
            # If columns are missing, collect the errors
            error_indices = set()
            error_reasons = {}
            collect_validation_errors(df, missing_columns, error_indices, error_reasons)

            error_message = f"Missing columns: {', '.join(missing_columns)}"
            send_email(
                "patelmit640@gmail.com",
                subject="Data Validation Failed",
                body=f"Data validation failed with the following issues:\n\n{error_message}",
            )
            logger.error(f"Data validation failed:\n{error_message}")
            return False
    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        return False


def list_bucket_blobs(bucket_name: str) -> list:
    """
    Lists all blobs in a Google Cloud Storage bucket.
    """
    setup_gcp_credentials()
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()
        blob_names = [blob.name for blob in blobs]
        logger.info(f"Found {len(blob_names)} files in bucket '{bucket_name}'")
        return blob_names
    except Exception as e:
        logger.error(f"Error listing blobs in bucket '{bucket_name}': {e}")
        raise


def validate_file(bucket_name: str, blob_name: str) -> bool:
    """
    Validates a single file from the specified GCP bucket.

    Parameters:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob/file to validate.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    try:
        logger.info(f"Loading data from GCS: {blob_name}")
        df = load_bucket_data(bucket_name, blob_name)

        logger.info(f"Validating data format for file: {blob_name}")
        validation_result = validate_data(df)

        if validation_result:
            logger.info(f"Validation passed for file: {blob_name}")
            return True
        else:
            logger.error(f"Validation failed for file: {blob_name}")
            return False

    except Exception as e:
        logger.error(f"Error validating file {blob_name}: {e}")
        return False


def main(cloud: bool = False, bucket_name: str = "full-raw-data"):
    """
    Main function to run the validation workflow on all files in a bucket.

    Parameters:
        cloud (bool): Whether to use cloud storage or local files.
        bucket_name (str): Name of the GCS bucket to validate files from.
    """
    try:
        if cloud:
            # Process all files in the bucket
            blob_names = list_bucket_blobs(bucket_name)
            if not blob_names:
                logger.warning(f"No files found in bucket '{bucket_name}'")
                return False

            all_valid = True
            for blob_name in blob_names:
                logger.info(f"Validating file: {blob_name}")
                file_valid = validate_file(bucket_name, blob_name)
                # Update overall validation status
                if not file_valid:
                    all_valid = False

            if all_valid:
                logger.info("All files in the bucket passed validation.")
                return True
            else:
                logger.error("One or more files failed validation.")
                return False
        else:
            # Local file validation (original behavior)
            file_name = "messy_transactions_20190103_20241231.xlsx"
            df = load_data(file_name)

            if not validate_data(df):
                logger.error("Validation failed. Exiting process.")
                return False

            logger.info("Workflow completed successfully.")
            return True

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run pre-validation on data")
    parser.add_argument(
        "--cloud", action="store_true", help="Load data from GCP bucket"
    )
    parser.add_argument(
        "--bucket", type=str, default="full-raw-data", help="GCP bucket name"
    )
    args = parser.parse_args()

    # Exit with non-zero code if validation fails
    success = main(cloud=args.cloud, bucket_name=args.bucket)
    if not success:
        print("Validation failed")
        exit(1)
    else:
        print("Validation successful")
        exit(0)
