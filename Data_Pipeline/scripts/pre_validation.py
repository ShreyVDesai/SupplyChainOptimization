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


def delete_blob_from_bucket(bucket_name: str, blob_name: str) -> bool:
    """
    Deletes a blob from a Google Cloud Storage bucket. Returns True if successful.
    """
    setup_gcp_credentials()
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Blob {blob_name} deleted from bucket {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting blob {blob_name} from bucket {bucket_name}: {e}")
        return False


def validate_file(
    bucket_name: str, blob_name: str, delete_invalid: bool = True
) -> bool:
    """
    Validates a single file from the specified GCP bucket.
    Deletes the file if validation fails and delete_invalid is True.

    Parameters:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob/file to validate.
        delete_invalid (bool): Whether to delete files that fail validation.

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

            # Delete the file if it failed validation and deletion is enabled
            if delete_invalid:
                logger.info(
                    f"Deleting invalid file: {blob_name} from bucket {bucket_name}"
                )
                delete_success = delete_blob_from_bucket(bucket_name, blob_name)
                if delete_success:
                    logger.info(f"Successfully deleted invalid file: {blob_name}")
                else:
                    logger.warning(f"Failed to delete invalid file: {blob_name}")

            return False

    except Exception as e:
        logger.error(f"Error validating file {blob_name}: {e}")

        # Delete the file if exception occurred during validation and deletion is enabled
        if delete_invalid:
            logger.info(
                f"Deleting file that caused exception: {blob_name} from bucket {bucket_name}"
            )
            delete_success = delete_blob_from_bucket(bucket_name, blob_name)
            if delete_success:
                logger.info(
                    f"Successfully deleted file that caused exception: {blob_name}"
                )
            else:
                logger.warning(
                    f"Failed to delete file that caused exception: {blob_name}"
                )

        return False


def main(
    cloud: bool = False, bucket_name: str = "full-raw-data", delete_invalid: bool = True
):
    """
    Main function to run the validation workflow on all files in a bucket.

    Parameters:
        cloud (bool): Whether to use cloud storage or local files.
        bucket_name (str): Name of the GCS bucket to validate files from.
        delete_invalid (bool): Whether to delete files that fail validation.

    Returns:
        int: 0 = all files valid, 1 = some files invalid but some valid, 2 = all files invalid or no files
    """
    try:
        if cloud:
            # Process all files in the bucket
            blob_names = list_bucket_blobs(bucket_name)
            if not blob_names:
                logger.warning(f"No files found in bucket '{bucket_name}'")
                print("No files found in bucket")
                return 2  # No files to process

            initial_file_count = len(blob_names)
            valid_files = []
            invalid_files = []

            for blob_name in blob_names:
                logger.info(f"Validating file: {blob_name}")
                file_valid = validate_file(bucket_name, blob_name, delete_invalid)
                if file_valid:
                    valid_files.append(blob_name)
                else:
                    invalid_files.append(blob_name)

            # Check validation results
            if len(valid_files) == initial_file_count:
                logger.info("All files in the bucket passed validation.")
                print(f"All files valid: {len(valid_files)} files")
                return 0  # All files valid
            elif len(valid_files) > 0:
                logger.info(
                    f"{len(valid_files)}/{initial_file_count} files passed validation."
                )
                print(
                    f"Partial validation: {len(valid_files)}/{initial_file_count} files valid, {len(invalid_files)} files invalid"
                )
                return 1  # Some files valid, some invalid
            else:
                logger.error("All files failed validation.")
                print("All files invalid")
                return 2  # All files invalid
        else:
            # Local file validation (original behavior)
            file_name = "messy_transactions_20190103_20241231.xlsx"
            df = load_data(file_name)

            if not validate_data(df):
                logger.error("Validation failed. Exiting process.")
                return 2

            logger.info("Workflow completed successfully.")
            return 0

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return 2


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run pre-validation on data")
    parser.add_argument(
        "--cloud", action="store_true", help="Load data from GCP bucket"
    )
    parser.add_argument(
        "--bucket", type=str, default="full-raw-data", help="GCP bucket name"
    )
    parser.add_argument(
        "--keep_invalid", action="store_true", help="Don't delete invalid files"
    )
    args = parser.parse_args()

    # Call main function with arguments
    status_code = main(
        cloud=args.cloud, bucket_name=args.bucket, delete_invalid=not args.keep_invalid
    )

    # Exit with appropriate code
    if status_code == 0:
        print("Validation successful")
        exit(0)
    elif status_code == 1:
        print("Validation partial - some files valid, some invalid")
        exit(1)  # We'll handle this special case in the DAG
    else:
        print("Validation failed - no valid files")
        exit(2)  # Critical failure
