import sys

sys.tracebacklimit = None
#!/usr/bin/env python
"""
DVC Manager Script

This script handles Data Version Control operations for the supply chain optimization
project, tracking the processed data in the specified Google Cloud Storage bucket.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import logging
from google.cloud import storage
import traceback

try:
    from logger import logger
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger


def run_command(command, working_dir=None):
    """
    Run a shell command and log the output

    Args:
        command (list): Command to run as a list of strings
        working_dir (str, optional): Directory to run the command in

    Returns:
        int: Return code of the command
    """
    # If the command is a DVC command, use the one from our virtual environment
    if command[0] == "dvc":
        command[0] = "/tmp/dvc_env/bin/dvc"

    logger.info(f"Running command: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=working_dir,
    )

    # Stream output in real-time
    for line in process.stdout:
        line = line.strip()
        if line:
            logger.info(line)

    # Get any error output
    stderr = process.communicate()[1]
    if stderr:
        for line in stderr.split("\n"):
            if line.strip():
                logger.warning(line)

    return process.returncode


def init_dvc(data_dir, remote_name, remote_url):
    """
    Initialize DVC in the specified directory

    Args:
        data_dir (str): Directory containing the data to version
        remote_name (str): Name for the DVC remote
        remote_url (str): URL for the DVC remote (GCS bucket)

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        data_path = Path(data_dir)

        # Create directory if it doesn't exist
        if not data_path.exists():
            data_path.mkdir(parents=True)
            logger.info(f"Created directory: {data_dir}")

        # Change to data directory
        os.chdir(data_dir)
        logger.info(f"Changed working directory to: {data_dir}")

        # Check if DVC is already initialized
        if Path(".dvc").exists():
            logger.info("DVC already initialized in this directory")
        else:
            # Initialize DVC
            if run_command(["dvc", "init", "--no-scm"]) != 0:
                logger.error("Failed to initialize DVC")
                return False
            logger.info("DVC initialized successfully")

        # Check if remote already exists
        config_file = Path(".dvc/config")
        if config_file.exists() and remote_name in config_file.read_text():
            logger.info(f"Remote '{remote_name}' already exists")
        else:
            # Configure DVC to use GCS
            if (
                run_command(["dvc", "remote", "add", "-d", remote_name, remote_url])
                != 0
            ):
                logger.error(f"Failed to add remote: {remote_url}")
                return False
            logger.info(f"DVC remote added: {remote_name} at {remote_url}")

        # Configure GCP credentials
        gcp_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if gcp_creds_path:
            if (
                run_command(
                    [
                        "dvc",
                        "remote",
                        "modify",
                        remote_name,
                        "credentialpath",
                        gcp_creds_path,
                    ]
                )
                != 0
            ):
                logger.warning("Failed to set GCP credentials path")

        return True

    except Exception as e:
        logger.error(f"Error initializing DVC: {str(e)}")
        return False


def add_and_push_data(data_dir, data_patterns=None, message=None):
    """
    Add data to DVC tracking and push to remote
    """
    try:
        logger.info(f"Starting DVC add and push operation in {data_dir}")

        # Change to data directory
        os.chdir(data_dir)
        logger.info(f"Changed working directory to: {data_dir}")

        # Default to all supported file types if no patterns specified
        if not data_patterns:
            data_patterns = ["*.csv", "*.xlsx", "*.xls", "*.json"]
            logger.info(f"Using default file patterns: {data_patterns}")
        else:
            logger.info(f"Using specified file patterns: {data_patterns}")

        # Remove existing .dvc files first
        dvc_files_removed = 0
        for dvc_file in Path(data_dir).glob("*.dvc"):
            if dvc_file.is_file():
                logger.info(f"Removing existing DVC file: {dvc_file}")
                dvc_file.unlink()
                dvc_files_removed += 1
        if dvc_files_removed > 0:
            logger.info(f"Removed {dvc_files_removed} existing .dvc files")

        # Track files matching patterns
        files_added = 0
        files_failed = 0
        total_size = 0

        for pattern in data_patterns:
            if pattern == ".dvc" or pattern.endswith(".dvc"):
                continue

            matching_files = list(Path(data_dir).glob(pattern))
            if not matching_files:
                logger.warning(f"No files match pattern: {pattern}")
                continue

            logger.info(
                f"Found {len(matching_files)} files matching pattern: {pattern}"
            )

            for file_path in matching_files:
                if (
                    file_path.is_file()
                    and not str(file_path).endswith(".dvc")
                    and not str(file_path.name).startswith(".")
                ):
                    try:
                        # Check if file is empty
                        file_size = file_path.stat().st_size
                        if file_size == 0:
                            logger.warning(f"Skipping empty file: {file_path.name}")
                            continue

                        logger.info(
                            f"Adding file to DVC: {file_path.name} ({file_size} bytes)"
                        )
                        result = run_command(["dvc", "add", str(file_path.name)])

                        if result == 0:
                            files_added += 1
                            total_size += file_size
                            logger.info(f"Successfully added: {file_path.name}")
                        else:
                            files_failed += 1
                            logger.error(
                                f"Failed to add file: {file_path.name}, error code: {result}"
                            )

                    except Exception as e:
                        files_failed += 1
                        logger.error(f"Error processing {file_path.name}: {str(e)}")

        # Log summary of files added
        logger.info(
            f"DVC add summary: {files_added} files added successfully, {files_failed} failed"
        )
        logger.info(f"Total data size: {total_size / (1024*1024):.2f} MB")

        if files_added == 0:
            logger.warning("No files were added to DVC tracking")
            return False

        # Push to DVC remote
        logger.info("Pushing data to DVC remote...")
        push_result = run_command(["dvc", "push"])

        if push_result != 0:
            logger.error("Failed to push data to DVC remote")
            return False

        logger.info(f"Successfully pushed {files_added} files to DVC remote")
        return True

    except Exception as e:
        logger.error(f"Error in DVC add/push operations: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False


def sync_from_gcs_to_workspace(source_bucket, workspace_dir):
    """
    Synchronize data from a GCS bucket to a local workspace for DVC tracking
    """
    try:
        logger.info(
            f"Starting sync from GCS bucket '{source_bucket}' to '{workspace_dir}'"
        )

        # Ensure workspace directory exists
        workspace_path = Path(workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        os.chdir(workspace_dir)

        # Clear existing files but preserve .dvc directory
        logger.info("Cleaning workspace directory...")
        files_removed = 0
        for item in workspace_path.iterdir():
            if item.is_file() and not item.name.endswith(".dvc"):
                item.unlink()
                files_removed += 1
            elif item.is_dir() and item.name != ".dvc" and item.name != ".git":
                shutil.rmtree(item)
                files_removed += 1
        logger.info(f"Removed {files_removed} existing items from workspace")

        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(source_bucket)

        # List all blobs in the bucket
        logger.info(f"Listing contents of bucket '{source_bucket}'...")
        blobs = list(bucket.list_blobs())

        if not blobs:
            logger.warning(f"No files found in bucket '{source_bucket}'")
            # Return True even when no files - it's not an error condition
            return True

        # Filter for supported file types
        supported_extensions = (".json", ".csv", ".xlsx", ".xls")
        valid_blobs = [
            b for b in blobs if b.name.lower().endswith(supported_extensions)
        ]

        if not valid_blobs:
            logger.warning(
                f"No supported files found in bucket. Supported types: {supported_extensions}"
            )
            # Return True even when no supported files - it's not an error condition
            return True

        logger.info(f"Found {len(valid_blobs)} supported files to sync")

        # Download and verify each file
        successful_downloads = 0
        failed_downloads = 0

        for blob in valid_blobs:
            dest_path = workspace_path / blob.name
            try:
                logger.info(f"Downloading: {blob.name}")
                blob.download_to_filename(dest_path)

                # Verify file integrity
                if not dest_path.exists():
                    logger.error(f"Failed to create file: {dest_path}")
                    failed_downloads += 1
                    continue

                if dest_path.stat().st_size == 0:
                    logger.error(f"Downloaded empty file: {blob.name}")
                    dest_path.unlink()  # Remove empty file
                    failed_downloads += 1
                    continue

                if dest_path.stat().st_size != blob.size:
                    logger.error(
                        f"Size mismatch for {blob.name}: expected {blob.size}, got {dest_path.stat().st_size}"
                    )
                    dest_path.unlink()
                    failed_downloads += 1
                    continue

                successful_downloads += 1
                logger.info(
                    f"Successfully downloaded and verified: {blob.name} ({dest_path.stat().st_size} bytes)"
                )

            except Exception as e:
                logger.error(f"Error downloading {blob.name}: {str(e)}")
                if dest_path.exists():
                    dest_path.unlink()
                failed_downloads += 1

        # Log summary
        logger.info(
            f"Sync completed: {successful_downloads} files downloaded successfully, {failed_downloads} failed"
        )

        # Empty source is not an error
        return True

    except Exception as e:
        logger.error(f"Error during sync operation: {str(e)}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        return False


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="DVC Manager for Supply Chain Optimization"
    )
    parser.add_argument(
        "--action",
        choices=["init", "sync", "add", "push", "full"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--data-dir",
        default="dvc_workspace",
        help="Directory to store version-controlled data",
    )
    parser.add_argument(
        "--source-bucket",
        default="fully-processed-data",
        help="GCS bucket to sync data from",
    )
    parser.add_argument(
        "--remote-name", default="gcp-remote", help="Name for DVC remote"
    )
    parser.add_argument(
        "--remote-url",
        default="gs://supply-chain-dvc-storage",
        help="GCS URL for DVC remote",
    )
    parser.add_argument(
        "--patterns", nargs="*", help="File patterns to add (default: all files)"
    )
    parser.add_argument("--message", help="Commit message (used for logging only)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (Data_Pipeline)
    parent_dir = os.path.dirname(script_dir)
    # Construct absolute path to workspace
    args.data_dir = os.path.join(parent_dir, args.data_dir)

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.debug("Verbose logging enabled")
        logger.debug(f"Using data directory: {args.data_dir}")

    # Execute the requested action
    if args.action == "init":
        success = init_dvc(args.data_dir, args.remote_name, args.remote_url)

    elif args.action == "sync":
        success = sync_from_gcs_to_workspace(args.source_bucket, args.data_dir)

    elif args.action == "add":
        success = add_and_push_data(args.data_dir, args.patterns, args.message)

    elif args.action == "push":
        os.chdir(args.data_dir)
        success = run_command(["dvc", "push"]) == 0

    elif args.action == "full":
        # Full pipeline: init → sync → add → push
        logger.info("Starting full DVC pipeline")

        logger.info("Step 1: Initializing DVC")
        if not init_dvc(args.data_dir, args.remote_name, args.remote_url):
            logger.error("Failed to initialize DVC")
            sys.exit(1)

        logger.info("Step 2: Syncing from GCS")
        sync_result = sync_from_gcs_to_workspace(args.source_bucket, args.data_dir)
        if not sync_result:
            logger.error("Failed to sync from GCS")
            sys.exit(1)

        # Check if there are any files to add after syncing
        workspace_path = Path(args.data_dir)
        supported_extensions = (".json", ".csv", ".xlsx", ".xls")
        files_to_add = []

        for ext in supported_extensions:
            files_to_add.extend(list(workspace_path.glob(f"*{ext}")))

        if not files_to_add:
            logger.warning(
                "No files to add after syncing. DVC add step will be skipped."
            )
            logger.info("Full DVC pipeline completed successfully (no files to track)")
            sys.exit(0)

        logger.info("Step 3: Adding and pushing data")
        if not add_and_push_data(args.data_dir, args.patterns, args.message):
            logger.error("Failed to add and push data")
            sys.exit(1)

        logger.info("Full DVC pipeline completed successfully")
        success = True

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Add missing import for datetime when running as main
    from datetime import datetime

    main()
