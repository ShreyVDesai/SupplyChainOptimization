import os
import pandas as pd
import json
from google.cloud import storage
from google.api_core.exceptions import Conflict


def excel_to_json(file_path):
    """
    Reads an Excel file and converts it to JSON format with ISO-formatted dates.

    Args:
        file_path (str): The local path to the Excel file.

    Returns:
        str: JSON string representation of the Excel data with ISO-formatted dates.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Convert the DataFrame to a JSON string with ISO date formatting
    json_str = df.to_json(orient="records", date_format="iso")

    return json_str


def pretty_print_json(json_str, num_records=5):
    """
    Pretty-prints a sample of the JSON data for verification.

    Args:
        json_str (str): JSON string representation of the data.
        num_records (int): Number of records to display for verification.
    """
    # Load the JSON string into a Python object
    json_obj = json.loads(json_str)

    # Ensure num_records does not exceed the length of the JSON object
    num_records = min(num_records, len(json_obj))

    # Pretty-print the specified number of records
    print(json.dumps(json_obj[:num_records], indent=4))


def upload_json_to_gcs(bucket_name, json_data, destination_blob_name):
    """
    Uploads JSON data to a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        json_data (str): The JSON data to upload.
        destination_blob_name (str): The destination blob name in the bucket.
    """
    # Initialize the GCS client
    storage_client = storage.Client()

    try:
        # Check if the bucket already exists
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except Exception:
        # If the bucket does not exist, create it
        print(f"Bucket '{bucket_name}' does not exist. Creating bucket...")
        try:
            bucket = storage_client.create_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        except Conflict:
            print(f"Bucket '{bucket_name}' already exists. Proceeding with upload.")
            bucket = storage_client.get_bucket(bucket_name)

    # Create a blob object
    blob = bucket.blob(destination_blob_name)

    # Upload the JSON data to GCS
    try:
        blob.upload_from_string(json_data, content_type="application/json")
        print(f"JSON data uploaded to bucket '{bucket_name}' as '{destination_blob_name}'.")
    except Exception as e:
        print(f"Error uploading JSON data: {e}")
        raise e


def process_all_excel_files_in_data_folder(data_folder):
    """
    Processes all Excel files in the specified folder, converts them to JSON,
    and uploads each to a GCS bucket.

    Args:
        data_folder (str): The path to the folder containing Excel files.
    """
    # Ensure the environment variable for authentication is set
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secret/gcp-key.json"

    # Iterate over all files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(data_folder, filename)

            # Convert Excel to JSON
            json_data = excel_to_json(file_path)

            # Pretty-print a sample of the JSON data for verification
            print(f"Sample of the converted JSON data from {filename}:")
            pretty_print_json(json_data)

            # Extract the bucket name from the Excel file name (without extension)
            bucket_name = os.path.splitext(filename)[0]

            # Define the destination blob name in the bucket
            destination_blob_name = bucket_name + ".json"

            # Upload the JSON data to GCS
            upload_json_to_gcs(bucket_name, json_data, destination_blob_name)


# Example usage
if __name__ == "__main__":
    data_folder = "data"  # Path to your data folder containing Excel files
    process_all_excel_files_in_data_folder(data_folder)
