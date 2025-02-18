import os
from google.cloud import storage
from dotenv import load_dotenv
import io
import polars as pl
from dataPreprocessing import *

# Load environment variables from .env file
load_dotenv()

def fetch_data_from_gcs(bucket_name, file_name):
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if not credentials_path:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set in .env file.")
        return

    storage_client = storage.Client.from_service_account_json(credentials_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob_content = blob.download_as_string()
    return pl.read_excel(io.BytesIO(blob_content))

# Example usage
bucket_name = 'mlops-data-storage-000'
file_name = 'generated_training_data/transactions_20190103_20241231.xlsx'
output_file = "cleaned_data.csv"

df = fetch_data_from_gcs(bucket_name, file_name)
df = clean_dates(df)
df = convert_feature_types(df)
df = fill_missing_value_with_unknown(df, ["Producer ID", "Store Location", "Transaction ID"])
# df = replace_digits_in_string_columns(df)
df = convert_string_columns_to_lowercase(df)
df = filling_missing_cost_price(df)
df = remove_invalid_records(df)
print("Saving Cleaned Data...")
bucket_name = 'mlops-data-storage-000'  # Replace with your GCS bucket name
destination_blob_name = 'cleaned_data/cleanedData_20190103_20241231.csv'  # GCS destination path
upload_df_to_gcs(bucket_name, df, destination_blob_name)
print(f"Cleaned data saved to {output_file}")

