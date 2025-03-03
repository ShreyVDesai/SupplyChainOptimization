#!/usr/bin/env python3
"""Debug script to check GCS bucket contents"""

import os
import sys
from google.cloud import storage
from dotenv import load_dotenv

# Try to import the setup_gcp_credentials function from the same place as the DVC script
try:
    from Data_Pipeline.scripts.utils import setup_gcp_credentials
except ImportError:
    # If not available, define a basic version
    def setup_gcp_credentials():
        """Basic GCP credential setup"""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            print(f"Using credentials from: {creds_path}")
        else:
            print(
                "No GOOGLE_APPLICATION_CREDENTIALS environment variable found"
            )
            gcp_key_paths = [
                "/app/secret/gcp-key.json",  # Path used in Docker container
                os.path.expanduser("~/.gcp/key.json"),
                "gcp-key.json",  # Current directory
            ]

            for path in gcp_key_paths:
                if os.path.exists(path):
                    print(f"Found credentials at: {path}")
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
                    return

            print("Warning: Could not find GCP credentials")


load_dotenv()


def list_bucket_files(bucket_name):
    """List all files in a GCS bucket with detailed info."""
    try:
        print(f"Listing files in bucket: {bucket_name}")

        # Setup GCP credentials
        setup_gcp_credentials()

        # Create a GCS client
        try:
            storage_client = storage.Client()
            print("Successfully created storage client")
        except Exception as e:
            print(f"Error creating storage client: {e}")
            return

        # Get the bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
            print(f"Successfully accessed bucket: {bucket_name}")
        except Exception as e:
            print(f"Error accessing bucket: {e}")
            return

        # List blobs/files
        try:
            blobs = list(bucket.list_blobs())

            if not blobs:
                print(f"No files found in bucket {bucket_name}")
                return

            print(f"Found {len(blobs)} files in bucket {bucket_name}:")

            for blob in blobs:
                print(
                    f"- {blob.name} (size: {blob.size} bytes, updated: {blob.updated})"
                )

        except Exception as e:
            print(f"Error listing blobs: {e}")

    except Exception as e:
        print(f"Error listing bucket contents: {e}")


def print_environment_info():
    """Print relevant environment information"""
    print("\nEnvironment Information:")
    print(
        f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}"
    )
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Available files in current directory: {os.listdir('.')}")


if __name__ == "__main__":
    print("Starting GCS bucket debug script")
    print_environment_info()

    print("\n--- Cache Bucket ---")
    list_bucket_files("fully-processed-cache")

    print("\n--- DVC Remote Bucket ---")
    list_bucket_files("fully-processed-data-dvc")
