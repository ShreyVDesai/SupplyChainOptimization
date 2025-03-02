import argparse

import polars as pl

try:
    from logger import logger
    from utils import (
        collect_validation_errors,
        delete_blob_from_bucket,
        list_bucket_blobs,
        load_bucket_data,
        send_email,
    )
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger
    from Data_Pipeline.scripts.utils import (
        send_email,
        load_bucket_data,
        list_bucket_blobs,
        delete_blob_from_bucket,
        collect_validation_errors,
    )

from dotenv import load_dotenv
from google.cloud import storage

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


def validate_data(df, filename=None):
    """
    Validate the DataFrame by checking if it contains all required columns.

    Parameters:
        df: DataFrame to validate
        filename (str, optional): Name of the file being validated, to include in error messages

    Returns:
        tuple: (is_valid, error_message) where:
            - is_valid (bool): True if all checks pass, False otherwise
            - error_message (str): Description of validation errors if any, None otherwise
    """
    try:
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Check if DataFrame is empty
        if len(df) == 0:
            error_message = "DataFrame is empty, no rows found"
            file_info = f" for file: {filename}" if filename else ""
            logger.error(
                f"Data validation failed{file_info}:\n{error_message}"
            )
            return False, error_message

        # Check if all required columns are present
        missing_columns = [
            col for col in PRE_VALIDATION_COLUMNS if col not in df.columns
        ]

        if not missing_columns:
            logger.info("Data validation passed successfully.")
            return True, None
        else:
            # If columns are missing, collect the errors
            error_indices = set()
            error_reasons = {}
            collect_validation_errors(
                df, missing_columns, error_indices, error_reasons
            )

            error_message = f"Missing columns: {', '.join(missing_columns)}"
            file_info = f" for file: {filename}" if filename else ""
            logger.error(
                f"Data validation failed{file_info}:\n{error_message}"
            )
            return False, error_message
    except Exception as e:
        file_info = f" for file: {filename}" if filename else ""
        error_message = str(e)
        logger.error(f"Error in data validation{file_info}: {e}")
        return False, error_message


def validate_file(
    bucket_name: str, blob_name: str, delete_invalid: bool = True
) -> tuple:
    """
    Validates a single file from the specified GCP bucket.
    Deletes the file if validation fails and delete_invalid is True.

    Parameters:
        bucket_name (str): Name of the GCS bucket.
        blob_name (str): Name of the blob/file to validate.
        delete_invalid (bool): Whether to delete files that fail validation.

    Returns:
        tuple: (is_valid, error_info) where:
            - is_valid (bool): True if validation passes, False otherwise
            - error_info (dict): Dictionary with error details if validation fails, None otherwise
    """
    try:
        logger.info(f"Loading data from GCS: {blob_name}")
        df = load_bucket_data(bucket_name, blob_name)

        logger.info(f"Validating data format for file: {blob_name}")
        validation_result, error_message = validate_data(
            df, filename=blob_name
        )

        if validation_result:
            logger.info(f"Validation passed for file: {blob_name}")
            return True, None
        else:
            logger.error(f"Validation failed for file: {blob_name}")
            error_info = {"filename": blob_name, "error": error_message}

            # Delete the file if it failed validation and deletion is enabled
            if delete_invalid:
                logger.info(
                    f"Deleting invalid file: {blob_name} from bucket {bucket_name}"
                )
                delete_success = delete_blob_from_bucket(
                    bucket_name, blob_name
                )
                if delete_success:
                    logger.info(
                        f"Successfully deleted invalid file: {blob_name}"
                    )
                else:
                    logger.warning(
                        f"Failed to delete invalid file: {blob_name}"
                    )
                    error_info["deletion_failed"] = True

            return False, error_info

    except Exception as e:
        logger.error(f"Error validating file {blob_name}: {e}")
        error_info = {
            "filename": blob_name,
            "error": f"Exception during validation: {str(e)}",
        }

        # Delete the file if exception occurred during validation and deletion
        # is enabled
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
                error_info["deletion_failed"] = True

        return False, error_info


def main(bucket_name: str = "full-raw-data", delete_invalid: bool = True):
    """
    Main function to run the validation workflow on all files in a bucket.

    Parameters:
        bucket_name (str): Name of the GCS bucket to validate files from.
        delete_invalid (bool): Whether to delete files that fail validation.

    Returns:
        int: 0 = all files valid, 1 = some files invalid but some valid, 2 = all files invalid or no files
    """
    try:
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
            file_valid, error_info = validate_file(
                bucket_name, blob_name, delete_invalid
            )
            if file_valid:
                valid_files.append(blob_name)
            else:
                invalid_files.append(error_info)

        # Send an aggregated email if there are any invalid files
        if invalid_files:
            email_subject = (
                f"Data Validation Failed for {len(invalid_files)} Files"
            )

            # Build the email body with information about each failed file
            email_body = f"Data validation failed for {len(invalid_files)} out of {initial_file_count} files:\n\n"

            for i, error_info in enumerate(invalid_files, 1):
                filename = error_info.get("filename", "Unknown file")
                error_message = error_info.get("error", "Unknown error")
                deletion_failed = error_info.get("deletion_failed", False)

                deletion_status = (
                    " (Deletion failed)" if deletion_failed else ""
                )
                email_body += (
                    f"{i}. {filename}{deletion_status}: {error_message}\n\n"
                )

            # Add summary of valid files
            if valid_files:
                email_body += (
                    f"\nSuccessfully validated {len(valid_files)} files:\n"
                )
                for i, valid_file in enumerate(valid_files, 1):
                    email_body += f"{i}. {valid_file}\n"

            # Send the aggregated validation report
            send_email(
                "talksick530@gmail.com",
                subject=email_subject,
                body=email_body,
            )
            logger.info(
                f"Sent aggregated validation report email for {len(invalid_files)} invalid files"
            )

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
    except Exception as e:
        logger.error(f"Error in main validation function: {e}")

        # Send an email about the failure in the validation process itself
        send_email(
            "talksick530@gmail.com",
            subject="Fatal Error in Data Validation Process",
            body=f"The validation process encountered a critical error:\n\n{str(e)}",
        )
        print(f"Error: {e}")
        return 2  # Error during process


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run pre-validation on data")
    parser.add_argument(
        "--bucket", type=str, default="full-raw-data", help="GCP bucket name"
    )
    parser.add_argument(
        "--keep_invalid",
        action="store_true",
        help="Don't delete invalid files",
    )
    args = parser.parse_args()

    # Call main function with arguments
    status_code = main(
        bucket_name=args.bucket, delete_invalid=not args.keep_invalid
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
