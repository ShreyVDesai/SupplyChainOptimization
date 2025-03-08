#!/usr/bin/env python3
"""
DVC Versioning Script

This script tracks processed data files using DVC without Git integration.
It tracks changes in the specified bucket directly and stores versioning information in a DVC remote.
"""

import os
import sys
import tempfile
import argparse
import subprocess
from typing import Optional, Tuple

try:
    from logger import logger
    from utils import setup_gcp_credentials
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger
    from Data_Pipeline.scripts.utils import setup_gcp_credentials

from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the DVC versioning script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Track processed files with DVC without Git"
    )

    parser.add_argument(
        "--destination_bucket",
        type=str,
        required=True,
        help="Destination bucket with processed data to track",
    )

    parser.add_argument(
        "--dvc_remote",
        type=str,
        required=True,
        help="DVC remote name for versioning",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def run_command(command: str, cwd: Optional[str] = None) -> Tuple[bool, str]:
    """
    Execute a shell command and return the result.

    Args:
        command (str): The command to execute
        cwd (str, optional): The working directory to execute the command in

    Returns:
        Tuple[bool, str]: A tuple containing success status and output/error message
    """
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}")
            logger.error(f"Command error: {result.stderr}")
            return False, result.stderr

        return True, result.stdout.strip()
    except Exception as e:
        logger.error(f"Exception while running command: {e}")
        return False, str(e)


def ensure_bucket_exists(bucket_name: str) -> bool:
    """
    Check if a GCS bucket exists and create it if it doesn't.

    Args:
        bucket_name (str): Name of the GCS bucket to check/create

    Returns:
        bool: True if bucket exists or was created successfully, False otherwise
    """
    try:
        logger.info(f"Checking if bucket {bucket_name} exists")

        # Setup Google Cloud credentials
        setup_gcp_credentials()

        # Create a GCS client
        storage_client = storage.Client()

        # Check if bucket exists
        try:
            storage_client.get_bucket(bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
            return True
        except Exception as e:
            if "Not Found" in str(e):
                logger.warning(f"Bucket {bucket_name} not found, creating it")
                try:
                    # Create the bucket
                    storage_client.create_bucket(bucket_name)
                    logger.info(f"Successfully created bucket {bucket_name}")
                    return True
                except Exception as create_e:
                    logger.error(
                        f"Failed to create bucket {bucket_name}: {create_e}"
                    )
                    return False
            else:
                logger.error(f"Error accessing bucket {bucket_name}: {e}")
                return False

    except Exception as e:
        logger.error(f"Unexpected error checking/creating bucket: {e}")
        return False


def list_bucket_files(bucket_name: str) -> dict:
    """
    List all files in a GCS bucket with their metadata.

    Args:
        bucket_name (str): Name of the GCS bucket

    Returns:
        dict: Dictionary of filename -> metadata (size, updated timestamp)
    """
    try:
        logger.info(f"Listing files in bucket {bucket_name}")

        # Setup Google Cloud credentials
        setup_gcp_credentials()

        # Create a GCS client
        storage_client = storage.Client()

        try:
            bucket = storage_client.get_bucket(bucket_name)
            blobs = list(bucket.list_blobs())

            # Create a dictionary of filename -> metadata
            files_metadata = {}
            for blob in blobs:
                files_metadata[blob.name] = {
                    "size": blob.size,
                    "updated": blob.updated,
                    "md5_hash": blob.md5_hash,
                }

            logger.info(
                f"Found {len(files_metadata)} files in bucket {bucket_name}"
            )
            return files_metadata

        except Exception as e:
            logger.error(f"Failed to list files in bucket {bucket_name}: {e}")
            return {}

    except Exception as e:
        logger.error(f"Unexpected error listing bucket files: {e}")
        return {}


def track_bucket_data(destination_bucket: str, dvc_remote: str) -> bool:
    """
    Track processed data with DVC without Git integration, working directly with remote storage.
    This avoids downloading all data from the bucket.

    Args:
        destination_bucket (str): Bucket with processed data to track
        dvc_remote (str): Name for the DVC remote

    Returns:
        bool: True if tracking succeeded, False otherwise
    """
    # Ensure all required buckets exist
    logger.info("Ensuring required buckets exist")
    if not ensure_bucket_exists(destination_bucket):
        logger.error(
            f"Failed to ensure destination bucket {destination_bucket} exists"
        )
        return False

    if not ensure_bucket_exists(dvc_remote):
        logger.error(f"Failed to ensure DVC remote bucket {dvc_remote} exists")
        return False

    # Get a list of files that exist before tracking
    # This will be used to detect what files were changed/overwritten
    logger.info(f"Getting baseline of existing files in {destination_bucket}")
    before_files = list_bucket_files(destination_bucket)

    # Create a temporary directory for DVC operations
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Working in temporary directory: {temp_dir}")

        # Ensure GCP credentials are properly set up
        setup_gcp_credentials()

        # Initialize DVC without Git
        logger.info("Initializing DVC")
        success, output = run_command("dvc init --no-scm -f", cwd=temp_dir)
        if not success:
            logger.error(f"Failed to initialize DVC: {output}")
            return False

        # Configure DVC to use GCP credentials properly
        logger.info("Configuring DVC for Google Cloud Storage")
        success, output = run_command(
            "dvc config core.no_scm true", cwd=temp_dir
        )
        if not success:
            logger.error(f"Failed to configure DVC: {output}")
            return False

        # Add DVC remote for versioning/cache
        logger.info(f"Adding cache remote: {dvc_remote} -> gs://{dvc_remote}")
        success, output = run_command(
            f"dvc remote add -d {dvc_remote} gs://{dvc_remote}", cwd=temp_dir
        )
        if not success:
            logger.error(f"Failed to add cache remote: {output}")
            return False

        # Configure remote to not require cloud authentication checks on each operation
        success, output = run_command(
            f"dvc remote modify {dvc_remote} checksum_jobs 1", cwd=temp_dir
        )
        if not success:
            logger.warning(
                f"Failed to optimize remote configuration: {output}"
            )
            # Continue as this is not critical

        # Use import-url to properly track the GCS bucket
        # This creates a proper .dvc file that can be tracked
        logger.info(f"Importing data from gs://{destination_bucket}")
        success, output = run_command(
            f"dvc import-url gs://{destination_bucket} {destination_bucket} --no-download",
            cwd=temp_dir,
        )
        if not success:
            # If empty bucket, handle specially
            if "URL does not exist" in output:
                logger.warning(
                    f"Bucket {destination_bucket} appears to be empty"
                )
                # Create a placeholder file to track the bucket
                logger.info("Creating empty bucket tracking file")
                placeholder_path = os.path.join(
                    temp_dir, f"{destination_bucket}.placeholder"
                )
                with open(placeholder_path, "w") as f:
                    f.write(
                        f"# Placeholder for tracking empty bucket: {destination_bucket}\n"
                    )

                # Add the placeholder file to DVC
                logger.info("Adding placeholder to DVC")
                success, output = run_command(
                    f"dvc add {destination_bucket}.placeholder", cwd=temp_dir
                )
                if not success:
                    logger.error(f"Failed to add placeholder: {output}")
                    return False
            else:
                logger.error(f"Failed to import bucket: {output}")
                return False

        # Get the files after tracking
        logger.info(f"Getting updated list of files in {destination_bucket}")
        after_files = list_bucket_files(destination_bucket)

        # Detect changes between before and after
        new_files = []
        updated_files = []

        for file_name, metadata in after_files.items():
            if file_name not in before_files:
                new_files.append(file_name)
            elif (
                before_files[file_name]["md5_hash"] != metadata["md5_hash"]
                or before_files[file_name]["size"] != metadata["size"]
                or before_files[file_name]["updated"] != metadata["updated"]
            ):
                updated_files.append(file_name)

        # Log the changes
        if new_files:
            logger.info(
                f"Detected {len(new_files)} new files added to bucket:"
            )
            for file_name in new_files:
                logger.info(f"  - New file: {file_name}")

        if updated_files:
            logger.info(
                f"Detected {len(updated_files)} files that were OVERWRITTEN:"
            )
            for file_name in updated_files:
                old_size = before_files[file_name]["size"]
                new_size = after_files[file_name]["size"]
                size_diff = new_size - old_size
                logger.info(
                    f"  - Overwritten: {file_name} (size change: {size_diff:+d} bytes)"
                )

        if not new_files and not updated_files:
            logger.info("No file changes detected in this run")

        # Push to DVC remote
        logger.info(f"Pushing tracked data to DVC remote: {dvc_remote}")
        success, output = run_command(f"dvc push", cwd=temp_dir)
        if not success:
            logger.error(f"Failed to push to remote: {output}")
            return False

        # Try to get DVC diff information if available
        success, output = run_command(f"dvc status --json", cwd=temp_dir)
        if success and output:
            logger.info(f"DVC status info: {output}")

        logger.info(
            f"Successfully tracked changes in bucket {destination_bucket} to DVC remote: {dvc_remote}"
        )
        return True


def main() -> None:
    """
    Main function to track processed data with DVC.
    """
    args = parse_arguments()

    # Configure logger verbosity
    if args.verbose:
        logger.setLevel("DEBUG")
        logger.info("Verbose logging enabled")

    # Get parameters from command line arguments
    dest_bucket = args.destination_bucket
    dvc_remote = args.dvc_remote

    # Log parameters for debugging
    logger.info(f"Destination bucket: {dest_bucket}")
    logger.info(f"DVC remote: {dvc_remote}")

    try:
        success = track_bucket_data(dest_bucket, dvc_remote)

        if success:
            logger.info("DVC versioning completed successfully")
            sys.exit(0)
        else:
            logger.error("DVC versioning failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception during DVC versioning: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
