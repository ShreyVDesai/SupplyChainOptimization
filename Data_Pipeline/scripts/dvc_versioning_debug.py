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
        "--cache_bucket",
        type=str,
        required=True,
        help="Cache bucket with processed data to track",
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
    Run a shell command and return its success status and output.

    Args:
        command (str): The command to run
        cwd (str, optional): Directory to run the command in

    Returns:
        Tuple[bool, str]: (Success status, Command output)
    """
    logger.debug(f"Running command: {command}")
    if cwd:
        logger.debug(f"Working directory: {cwd}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output = result.stdout
        error = result.stderr

        # Combine stdout and stderr for a complete log
        combined_output = output + ("\n" + error if error else "")

        if result.returncode != 0:
            logger.debug(f"Command failed with exit code: {result.returncode}")
            logger.debug(f"Command output: {combined_output}")
            return False, combined_output
        else:
            logger.debug(f"Command succeeded")
            logger.debug(f"Command output: {combined_output}")
            return True, combined_output

    except Exception as e:
        error_msg = f"Error executing command '{command}': {e}"
        logger.error(error_msg)
        return False, error_msg


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


def track_bucket_data(cache_bucket: str, dvc_remote: str) -> bool:
    """
    Track changes in a GCS bucket using DVC, without Git integration.
    This function detects changes, tracks them with DVC and pushes to a DVC remote.

    Args:
        cache_bucket (str): Bucket with processed data to track
        dvc_remote (str): DVC remote name for storing tracking data

    Returns:
        bool: True if tracking was successful, False otherwise
    """
    # Create a temporary directory for DVC operations
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        # Setup GCP credentials explicitly
        setup_gcp_credentials()

        # Ensure both buckets exist before proceeding
        logger.info("Ensuring cache bucket exists")
        if not ensure_bucket_exists(cache_bucket):
            logger.error(
                f"Failed to ensure cache bucket {cache_bucket} exists"
            )
            return False

        logger.info("Ensuring DVC remote bucket exists")
        if not ensure_bucket_exists(dvc_remote):
            logger.error(
                f"Failed to ensure DVC remote bucket {dvc_remote} exists"
            )
            return False

        # Initialize DVC in the temporary directory
        logger.info("Initializing DVC")
        success, output = run_command("dvc init --no-scm -f", cwd=temp_dir)
        if not success:
            logger.error(f"Failed to initialize DVC: {output}")
            return False

        # Configure DVC to use no SCM explicitly
        logger.info("Configuring DVC for no source control management")
        success, output = run_command(
            "dvc config core.no_scm true", cwd=temp_dir
        )
        if not success:
            logger.error(f"Failed to configure DVC no_scm: {output}")
            return False

        # Get baseline of existing files
        logger.info(f"Getting baseline of existing files in {cache_bucket}")
        before_files = list_bucket_files(cache_bucket)
        logger.debug(f"Files before: {before_files}")

        # Configure remote storage with explicit settings for GCS
        logger.info(f"Configuring DVC remote: {dvc_remote}")
        success, output = run_command(
            f"dvc remote add {dvc_remote} gs://{dvc_remote}",
            cwd=temp_dir,
        )
        if not success:
            logger.error(f"Failed to add DVC remote: {output}")
            return False

        # Configure the remote to use a higher number of threads for better performance
        logger.info("Optimizing DVC remote configuration")
        run_command(
            f"dvc remote modify {dvc_remote} checksum_jobs 16",
            cwd=temp_dir,
        )

        # Set the remote as default
        logger.info("Setting the remote as default")
        success, output = run_command(
            f"dvc remote default {dvc_remote}",
            cwd=temp_dir,
        )
        if not success:
            logger.error(f"Failed to set default remote: {output}")
            return False

        # Import data from bucket
        logger.info(f"Importing data from gs://{cache_bucket}")
        success, output = run_command(
            f"dvc import-url gs://{cache_bucket} {cache_bucket} --no-download",
            cwd=temp_dir,
        )

        # If bucket is empty, create a placeholder file to track
        if not success and "No URLs found" in output:
            logger.warning(f"Bucket {cache_bucket} appears to be empty")
            logger.info("Creating a placeholder file for tracking")

            # Create a placeholder file
            placeholder_path = os.path.join(
                temp_dir, f"{cache_bucket}.placeholder"
            )
            with open(placeholder_path, "w") as f:
                f.write(
                    f"# Placeholder for tracking empty bucket: {cache_bucket}\n"
                )

            # Add the placeholder to DVC
            logger.info("Adding placeholder to DVC")
            success, output = run_command(
                f"dvc add {cache_bucket}.placeholder", cwd=temp_dir
            )
            if not success:
                logger.error(f"Failed to add placeholder: {output}")
                return False
        elif not success:
            logger.error(f"Failed to import data: {output}")
            return False

        # Get updated list of files
        logger.info(f"Getting updated list of files in {cache_bucket}")
        after_files = list_bucket_files(cache_bucket)
        logger.debug(f"Files after: {after_files}")

        # Detect changes
        added = {k: v for k, v in after_files.items() if k not in before_files}
        removed = {
            k: v for k, v in before_files.items() if k not in after_files
        }
        modified = {
            k: after_files[k]
            for k in set(before_files) & set(after_files)
            if before_files[k] != after_files[k]
        }

        # Log changes
        if added or removed or modified:
            logger.info("Changes detected in the bucket")
            if added:
                logger.info(f"Added files: {list(added.keys())}")
            if removed:
                logger.info(f"Removed files: {list(removed.keys())}")
            if modified:
                logger.info(f"Modified files: {list(modified.keys())}")
        else:
            logger.info("No changes detected in the bucket")

        # Make sure all files are tracked by DVC
        logger.info("Ensuring all files are tracked by DVC")
        success, output = run_command("dvc add .", cwd=temp_dir)
        if not success:
            logger.warning(f"Issue with dvc add: {output}, but continuing...")

        # Explicitly commit the changes
        logger.info("Committing changes to DVC")
        success, output = run_command("dvc commit", cwd=temp_dir)
        if not success:
            logger.warning(
                f"Issue with dvc commit: {output}, but continuing..."
            )

        # Push all changes to remote with verbose output
        logger.info(f"Pushing changes to remote: {dvc_remote}")
        success, output = run_command("dvc push -v", cwd=temp_dir)
        if not success:
            logger.error(f"Failed to push changes: {output}")
            return False

        # Verify push completed
        logger.info("Verifying push to remote completed")
        success, output = run_command("dvc status -c", cwd=temp_dir)
        if not success:
            logger.warning(f"Issue checking status: {output}")
        else:
            logger.info(f"DVC status check: {output}")

        logger.info(
            f"Successfully tracked changes in bucket {cache_bucket} to DVC remote: {dvc_remote}"
        )
        return True
    except Exception as e:
        logger.error(f"Error during DVC tracking: {e}")
        return False
    finally:
        # Clean up temporary directory if it still exists
        import shutil

        try:
            if os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")


def main() -> None:
    """
    Main function to track processed data with DVC.
    """
    args = parse_arguments()

    # Configure logger verbosity
    if args.verbose:
        import logging; #logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    # Get parameters from command line arguments
    dest_bucket = args.cache_bucket
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
