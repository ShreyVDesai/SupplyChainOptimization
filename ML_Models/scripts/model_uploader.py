import os
import pickle  # Only needed if you're doing any in-memory pickle operations
from dotenv import load_dotenv
from google.cloud import storage

try:
    from logger import logger
except ImportError:  # Fallback if `logger` not found
    from Data_Pipeline.scripts.logger import logger

load_dotenv()

def upload_pickle_to_gcs(local_pickle_path: str, bucket_name: str, destination_blob_name: str) -> None:
    
    # Uploads a local pickle file to Google Cloud Storage (GCS).

    # Args:
    #     local_pickle_path (str): The path to the pickle file on local disk, e.g. "models/my_model.pkl".
    #     bucket_name (str): The name of the GCS bucket.
    #     destination_blob_name (str): The desired blob name/path in GCS, e.g. "trained_models/my_model.pkl".

    # Raises:
    #     Exception: If any error occurs during the process.
    
    # If you're using a custom method to set up local credentials, uncomment
    # your existing function if needed.
    # setup_gcp_credentials()

    try:
        logger.info(
            "Starting upload of pickle file to GCS. Bucket: %s, Blob: %s",
            bucket_name,
            destination_blob_name
        )

        # Initialize a storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Option A: Upload from local file path
        blob.upload_from_filename(local_pickle_path, content_type="application/octet-stream")

        # If you had the pickle file in memory, you could do:
        # with open(local_pickle_path, "rb") as f:
        #     blob.upload_from_file(f, content_type="application/octet-stream")

        logger.info("Pickle file uploaded successfully to gs://%s/%s", bucket_name, destination_blob_name)

    except Exception as e:
        logger.error("Error uploading pickle file to GCS. Error: %s", e)
        raise
