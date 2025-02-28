#!/usr/bin/env python3
"""
Test Pipeline Script

This script tests the data pipeline by executing the validation and preprocessing steps
in sequence, mimicking the Airflow DAG flow for local testing.

Usage:
    python test_pipeline.py <input_file_path> [<output_file_path>] [--cloud]

Example:
    python test_pipeline.py sample_data.csv output/cleaned_data.json
    python test_pipeline.py sample_data.csv output/cleaned_data.json --cloud
"""

import os
import sys
import logging
import importlib.util
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline_test.log")],
)
logger = logging.getLogger(__name__)


def import_script(script_path):
    """Import a Python script as a module."""
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_output_file(file_path, cloud_mode, bucket_name=None, blob_name=None):
    """Verify that the output file exists either locally or in the cloud."""
    if cloud_mode:
        if not bucket_name or not blob_name:
            logger.error("Bucket name and blob name required for cloud verification")
            return False

        try:
            # Import GCS libraries
            from google.cloud import storage

            # Check if the file exists
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if blob.exists():
                logger.info(
                    f"Verified output file exists in GCS: {bucket_name}/{blob_name}"
                )
                return True
            else:
                logger.error(
                    f"Output file does not exist in GCS: {bucket_name}/{blob_name}"
                )
                return False

        except Exception as e:
            logger.error(f"Error verifying cloud output file: {e}", exc_info=True)
            return False
    else:
        # Local file verification
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"Verified local output file exists: {file_path}")
            return True
        else:
            logger.error(f"Local output file missing or empty: {file_path}")
            return False


def run_pipeline(input_file, output_file=None, cloud_mode=False):
    """
    Run the complete data pipeline (validation + preprocessing).

    Args:
        input_file (str): Path to the input data file
        output_file (str, optional): Path to save the output file
        cloud_mode (bool): Whether to use cloud storage for output

    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    try:
        # Default output file if not provided
        if not output_file:
            base_dir = os.path.dirname(input_file)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(base_dir, f"{base_name}_cleaned.json")

        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Step 1: Import validation script
        logger.info(f"Importing validation script...")
        validation_path = os.path.join(script_dir, "DataValidation_Schema&Stats.py")
        validation_module = import_script(validation_path)

        # Step 2: Run validation
        logger.info(f"Running validation on {input_file}")
        validation_result = validation_module.analyze_data(input_file)

        if validation_result:
            logger.warning("Validation detected issues with the data")
            user_response = input(
                "Continue with preprocessing despite validation issues? (y/n): "
            )
            if user_response.lower() != "y":
                logger.info("Pipeline stopped after validation as per user request")
                return False
        else:
            logger.info("Validation completed successfully")

        # Step 3: Import preprocessing script
        logger.info("Importing preprocessing script...")
        preprocessing_path = os.path.join(script_dir, "dataPreprocessing.py")
        preprocessing_module = import_script(preprocessing_path)

        # Step 4: Set up paths and buckets
        source_bucket = "full-raw-data"
        destination_bucket = "fully-processed-data"

        # For cloud mode, use source blob path relative to input file
        source_blob = os.path.basename(input_file)
        dest_blob = f"cleaned_data/processed_{os.path.basename(input_file)}"
        # Make sure dest_blob has a .json extension
        dest_blob = f"{os.path.splitext(dest_blob)[0]}.json"

        # Step 5: Run preprocessing
        logger.info(f"Running preprocessing on {input_file} (cloud mode: {cloud_mode})")
        try:
            preprocessing_result = preprocessing_module.main(
                input_file=input_file,
                output_file=output_file,
                bucket_name=source_bucket,  # Source bucket name
                source_blob_name=source_blob,  # Source blob name
                destination_bucket_name=destination_bucket,  # Destination bucket name
                destination_blob_name=dest_blob,  # Destination blob name
                output_format="json",  # Use JSON format
                cloud=cloud_mode,  # Use mode specified by parameter
            )

            if not preprocessing_result:
                logger.error("Preprocessing reported failure")
                return False

            # Step 6: Verify output exists
            verification_result = verify_output_file(
                output_file,
                cloud_mode,
                destination_bucket if cloud_mode else None,
                dest_blob if cloud_mode else None,
            )

            if not verification_result:
                logger.error("Output verification failed - file not found or empty")
                return False

            logger.info(
                f"Pipeline completed successfully! Output saved to: {'GCS: ' + destination_bucket + '/' + dest_blob if cloud_mode else output_file}"
            )
            return True

        except Exception as preprocess_error:
            logger.error(f"Preprocessing failed: {preprocess_error}", exc_info=True)
            return False

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the data pipeline for testing")
    parser.add_argument("input_file", help="Path to the input data file")
    parser.add_argument(
        "output_file", nargs="?", help="Path for the output file (optional)"
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Use cloud storage for output"
    )

    args = parser.parse_args()

    # Run the pipeline
    success = run_pipeline(args.input_file, args.output_file, args.cloud)

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
