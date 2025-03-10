#!/usr/bin/env python3
"""
DVC Versioning Script for GCP Buckets

This script properly versions files in a GCP bucket using DVC.
It creates a proper DVC remote for versioning while maintaining a stateless approach.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from google.cloud import storage
from google.api_core.exceptions import NotFound

# Import common utilities
try:
    from logger import logger
    from utils import setup_gcp_credentials
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger
    from Data_Pipeline.scripts.utils import setup_gcp_credentials

# For filename quoting in shell commands
from shlex import quote as shell_quote


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Version files in a GCP bucket using DVC"
    )
    parser.add_argument(
        "--cache_bucket",
        required=True,
        help="Name of the GCP bucket containing files to version",
    )
    parser.add_argument(
        "--dvc_remote",
        required=True,
        help="Name of the GCP bucket for DVC remote storage",
    )
    parser.add_argument(
        "--clear_remote",
        action="store_true",
        help="Clear the remote DVC bucket before starting",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary directory for debugging",
    )
    parser.add_argument(
        "--gcp_key_path",
        default="/app/secret/gcp-key.json",
        help="Path to GCP service account key JSON file",
    )
    return parser.parse_args()


def run_command(
    command: str,
    cwd: Optional[str] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """
    Run a shell command and return its success status and output.

    Args:
        command: Command to run
        cwd: Directory to run command in
        check: Whether to raise an exception on command failure
        env: Environment variables to set

    Returns:
        Tuple of (success, output)
    """
    logger.info(f"Running command: {command}")

    # Create a copy of the current environment
    cmd_env = os.environ.copy()
    # Update with any additional environment variables
    if env:
        cmd_env.update(env)

    try:
        # Set PYTHONUNBUFFERED to get real-time output
        cmd_env["PYTHONUNBUFFERED"] = "1"

        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=cmd_env,
        )
        success = result.returncode == 0
        output = result.stdout.strip()

        if success:
            logger.info(f"Command completed successfully")
            if output and len(output) > 0:
                logger.debug(f"Output: {output}")
        else:
            logger.error(f"Command failed with exit code {result.returncode}")
            logger.error(f"Error output: {output}")

        return success, output
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {e}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)


def ensure_bucket_exists(bucket_name: str, gcp_key_path: str = None) -> bool:
    """
    Check if a GCP bucket exists and create it if it doesn't.

    Args:
        bucket_name: Name of the bucket to check
        gcp_key_path: Path to GCP key file

    Returns:
        True if bucket exists or was created, False otherwise
    """
    logger.info(f"Checking if bucket {bucket_name} exists")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        if bucket.exists():
            logger.info(f"Bucket {bucket_name} already exists")
            return True

        # Create bucket if it doesn't exist
        bucket = storage_client.create_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} created successfully")
        return True
    except Exception as e:
        logger.error(f"Error checking/creating bucket {bucket_name}: {e}")
        return False


def clear_bucket(bucket_name: str, gcp_key_path: str = None) -> bool:
    """
    Clear all objects from a GCP bucket.

    Args:
        bucket_name: Name of the bucket to clear
        gcp_key_path: Path to GCP key file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Clearing bucket {bucket_name}")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        if not bucket.exists():
            logger.warning(f"Bucket {bucket_name} does not exist")
            return True

        blobs = bucket.list_blobs()
        for blob in blobs:
            logger.debug(f"Deleting {blob.name}")
            blob.delete()

        logger.info(f"Bucket {bucket_name} cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Error clearing bucket {bucket_name}: {e}")
        return False


def list_bucket_files(bucket_name: str, gcp_key_path: str = None) -> List[str]:
    """
    List all files in a GCP bucket.

    Args:
        bucket_name: Name of the bucket to list files from
        gcp_key_path: Path to GCP key file

    Returns:
        List of file paths in the bucket
    """
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name)

        # Filter out directories or special files
        file_paths = [
            blob.name for blob in blobs if not blob.name.endswith("/")
        ]

        logger.info(f"Found {len(file_paths)} files in bucket {bucket_name}")
        return file_paths
    except Exception as e:
        logger.error(f"Error listing files in bucket {bucket_name}: {e}")
        return []


def list_dvc_remote_contents(
    dvc_remote: str, gcp_key_path: str = None
) -> List[str]:
    """
    List contents of the DVC remote bucket.

    Args:
        dvc_remote: Name of the DVC remote bucket
        gcp_key_path: Path to GCP key file

    Returns:
        List of files in the DVC remote bucket
    """
    try:
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(dvc_remote)
        files = [blob.name for blob in blobs]

        logger.info(f"DVC Remote Contents ({len(files)} files):")
        for file in files[:20]:  # Limit to first 20 files to avoid log spam
            logger.info(f"  - {file}")
        if len(files) > 20:
            logger.info(f"  ... and {len(files) - 20} more files")

        return files
    except Exception as e:
        logger.error(f"Error listing DVC remote contents: {e}")
        return []


def save_version_metadata(
    dvc_remote: str,
    cache_bucket: str,
    file_info: List[Dict[str, Any]],
    gcp_key_path: str = None,
) -> bool:
    """
    Save version metadata to the DVC remote bucket.

    Args:
        dvc_remote: Name of the DVC remote bucket
        cache_bucket: Name of the cache bucket
        file_info: List of file information dictionaries
        gcp_key_path: Path to GCP key file

    Returns:
        True if successful, False otherwise
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(dvc_remote)

        # Get existing metadata if it exists
        metadata_blob_name = f"dvc_metadata/{cache_bucket}_versions.json"
        blob = bucket.blob(metadata_blob_name)

        try:
            content = blob.download_as_text()
            metadata = json.loads(content)
        except (NotFound, Exception):
            metadata = {"versions": []}

        # Add new version information
        timestamp = datetime.now().isoformat()
        new_version = {
            "timestamp": timestamp,
            "files": file_info,
        }

        # Append new version to the list
        metadata["versions"].append(new_version)

        # Upload updated metadata to GCS
        blob.upload_from_string(json.dumps(metadata, indent=2))

        logger.info(f"Saved version metadata to {metadata_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving version metadata: {e}")
        return False


def setup_and_verify_dvc_remote(
    temp_dir: str,
    dvc_remote: str,
    gcp_key_path: str = None,
    debug_mode: bool = False,
) -> bool:
    """
    Set up and verify DVC remote configuration.

    Args:
        temp_dir: Directory where DVC is initialized
        dvc_remote: Name of the DVC remote bucket
        gcp_key_path: Path to GCP key file
        debug_mode: Whether to enable debug mode

    Returns:
        True if successful, False otherwise
    """
    logger.info("Setting up and verifying DVC remote")

    # Initialize DVC
    logger.info("Initializing DVC")
    success, output = run_command("dvc init --no-scm", cwd=temp_dir)
    if not success:
        logger.error(f"Failed to initialize DVC: {output}")
        return False

    # Set up DVC remote
    logger.info(f"Setting up DVC remote: gs://{dvc_remote}")
    remote_url = f"gs://{dvc_remote}"
    success, output = run_command(
        f"dvc remote add -d myremote {remote_url}", cwd=temp_dir
    )
    if not success:
        logger.error(f"Failed to add DVC remote: {output}")
        return False

    # Configure remote options
    run_command(f"dvc remote modify myremote checksum_jobs 16", cwd=temp_dir)
    run_command(f"dvc remote modify myremote jobs 4", cwd=temp_dir)

    # Add GCP credentials to DVC config
    if gcp_key_path and os.path.exists(gcp_key_path):
        logger.info(f"Adding Google credentials to DVC config: {gcp_key_path}")
        run_command(
            f"dvc remote modify --local myremote credentialpath {gcp_key_path}",
            cwd=temp_dir,
        )

    # In debug mode, dump the config
    if debug_mode:
        logger.info("DVC configuration:")
        success, output = run_command("cat .dvc/config", cwd=temp_dir)
        if success:
            logger.info(output)
        success, output = run_command("cat .dvc/config.local", cwd=temp_dir)
        if success:
            logger.info(output)

    # Test the remote connection with a simple file
    logger.info("Testing DVC remote connection")
    success, output = run_command(
        "touch test_file.txt && dvc add test_file.txt && dvc push -v",
        cwd=temp_dir,
    )

    # In debug mode, show what files were created
    if debug_mode:
        run_command("find .dvc -type f | sort", cwd=temp_dir)

    return success


def track_bucket_data(
    cache_bucket: str,
    dvc_remote: str,
    debug_mode: bool = False,
    keep_temp: bool = False,
    gcp_key_path: str = None,
    clear_remote: bool = False,
) -> bool:
    """
    Track files in a GCP bucket using DVC.

    Args:
        cache_bucket: Name of the bucket containing files to track
        dvc_remote: Name of the bucket to use for DVC remote storage
        debug_mode: Whether to enable debug mode
        keep_temp: Whether to keep the temporary directory
        gcp_key_path: Path to GCP key file
        clear_remote: Whether to clear the remote bucket before pushing

    Returns:
        True if successful, False otherwise
    """
    if debug_mode:
        logger.info(
            "DEBUG MODE ENABLED - Will use alternative approaches for troubleshooting"
        )

    logger.info(f"Cache bucket: {cache_bucket}")
    logger.info(f"DVC remote: {dvc_remote}")

    # Create a unique temporary directory to avoid conflicts
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")

    # Set GCP credentials using the utility function
    setup_gcp_credentials()

    try:
        # Ensure buckets exist
        logger.info("Ensuring buckets exist")
        for bucket in [cache_bucket, dvc_remote]:
            if not ensure_bucket_exists(bucket):
                logger.error(f"Failed to ensure bucket {bucket} exists")
                return False

        # Clear the remote bucket if requested
        if clear_remote:
            logger.info(f"Clearing remote bucket {dvc_remote}")
            if not clear_bucket(dvc_remote):
                logger.error(f"Failed to clear bucket {dvc_remote}")
                return False

        # List files in the cache bucket
        bucket_files = list_bucket_files(cache_bucket)
        if not bucket_files:
            logger.warning(f"No files found in bucket {cache_bucket}")
            # Still create version metadata with empty files list
            save_version_metadata(dvc_remote, cache_bucket, [])
            return True

        # Set up DVC with remote
        if not setup_and_verify_dvc_remote(
            temp_dir, dvc_remote, gcp_key_path, debug_mode
        ):
            logger.error("Failed to set up and verify DVC remote")
            return False

        # Download and track each file
        logger.info(
            f"Processing {len(bucket_files)} files from gs://{cache_bucket}"
        )

        file_info = []
        successful_imports = 0

        for file_path in bucket_files:
            logger.info(f"Processing file: {file_path}")

            try:
                # Get file metadata
                storage_client = storage.Client()
                bucket = storage_client.get_bucket(cache_bucket)
                blob = bucket.blob(file_path)
                blob.reload()

                # Create a safe filename without spaces or special characters
                original_name = os.path.basename(file_path)
                safe_name = original_name.replace(" ", "_")
                local_path = os.path.join(temp_dir, safe_name)

                logger.info(
                    f"Using safe filename: {safe_name} for original: {original_name}"
                )

                # Download the file locally first
                gs_url = f"gs://{cache_bucket}/{file_path}"
                logger.info(f"Downloading {gs_url} to {local_path}")

                # Use direct GCS download
                blob.download_to_filename(local_path)

                # Add and push the file with DVC
                logger.info(f"Adding {safe_name} to DVC")
                success, output = run_command(
                    f"dvc add {shell_quote(safe_name)}", cwd=temp_dir
                )

                if not success:
                    logger.error(f"Failed to add {safe_name} to DVC: {output}")
                    continue

                # Push immediately
                logger.info(f"Pushing {safe_name} to DVC remote")
                success, output = run_command(
                    f"dvc push -v {shell_quote(safe_name)}.dvc", cwd=temp_dir
                )

                if success:
                    logger.info(f"Successfully pushed {file_path}")
                    successful_imports += 1

                    # Add file information to our metadata
                    file_info.append(
                        {
                            "file_path": file_path,
                            "size": blob.size,
                            "md5": blob.md5_hash,
                            "updated": (
                                blob.updated.isoformat()
                                if blob.updated
                                else None
                            ),
                            "generation": blob.generation,
                        }
                    )
                else:
                    logger.error(f"Failed to push {file_path}: {output}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        # Check what was pushed to the remote
        time.sleep(2)  # Brief delay to allow GCS to update
        logger.info("Checking DVC remote contents after push")
        list_dvc_remote_contents(dvc_remote)

        # Save our own version metadata
        save_version_metadata(dvc_remote, cache_bucket, file_info)

        logger.info(
            f"Successfully versioned {successful_imports} out of {len(bucket_files)} files"
        )

        return successful_imports > 0 or len(bucket_files) == 0

    except Exception as e:
        logger.error(f"Error in track_bucket_data: {e}")
        return False
    finally:
        if not keep_temp:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            logger.info(
                f"Keeping temporary directory for debugging: {temp_dir}"
            )


def debug_dvc_setup(temp_dir: str):
    """
    Print debugging information about the DVC setup.

    Args:
        temp_dir: Path to the temporary DVC directory
    """
    logger.info("==== DVC DEBUGGING INFORMATION ====")

    # Check DVC config files
    run_command(
        "find .dvc -name '*.config' -o -name 'config*'",
        cwd=temp_dir,
        check=False,
    )
    logger.info("DVC config files:")

    # Display DVC config
    success, output = run_command("cat .dvc/config", cwd=temp_dir, check=False)
    logger.info("DVC config:")
    if success and output:
        logger.info(output)

    # Display local config if it exists
    success, output = run_command(
        "cat .dvc/config.local", cwd=temp_dir, check=False
    )
    logger.info("DVC local config:")
    if success and output:
        logger.info(output)

    # List all DVC files
    success, output = run_command(
        "find . -name '*.dvc'", cwd=temp_dir, check=False
    )
    logger.info("DVC files created:")
    if success and output:
        logger.info(output)

    # Show cache directory
    success, output = run_command(
        "find .dvc/cache -type f | head -10", cwd=temp_dir, check=False
    )
    logger.info("DVC cache directory:")
    if success and output:
        logger.info(output)

    # Show DVC status
    success, output = run_command("dvc status", cwd=temp_dir, check=False)
    logger.info("DVC status:")
    if success and output:
        logger.info(output)

    # Show remote status
    success, output = run_command(
        "dvc status -r myremote", cwd=temp_dir, check=False
    )
    logger.info("DVC remote status:")
    if success and output:
        logger.info(output)

    logger.info("====================================")


def main() -> None:
    """Main function to run the script."""
    args = parse_arguments()

    # Enable verbose logging if debug mode is enabled
    if args.debug:
        logger.setLevel("DEBUG")
        logger.info("Verbose logging enabled")

    # Track data in the GCP bucket
    success = track_bucket_data(
        args.cache_bucket,
        args.dvc_remote,
        args.debug,
        args.keep_temp,
        args.gcp_key_path,
        args.clear_remote,
    )

    if success:
        logger.info("DVC versioning completed successfully")
        sys.exit(0)
    else:
        logger.error("DVC versioning failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
