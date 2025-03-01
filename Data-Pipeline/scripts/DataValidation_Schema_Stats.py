import os
import polars as pl
import pandas as pd
from google.cloud import storage
import great_expectations as ge
import smtplib
from email.message import EmailMessage
import json
from logger import logger
from utils import send_email, load_bucket_data, load_data


# def fetch_file_from_gcp(bucket_name, file_name, destination):
#     """
#     Fetch file from the specified GCP bucket and save it to the destination.
    
#     Parameters:
#       bucket_name (str): Name of the GCP bucket.
#       file_name (str): Name/path of the file in the bucket.
#       destination (str): Local destination path.
#     """
#     try:
#         # Ensure the destination directory exists
#         os.makedirs(os.path.dirname(destination), exist_ok=True)
        
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         blob = bucket.blob(file_name)
#         blob.download_to_filename(destination)
#         logger.info(f"File {file_name} downloaded from GCP bucket {bucket_name}.")
#     except Exception as e:
#         logger.error(f"Error fetching file from GCP: {e}")
#         raise

# def load_data(file_path):
#     """
#     Load data from a CSV or Excel file into a Pandas DataFrame.
    
#     Parameters:
#       file_path (str): Local path to the file.
#     """
#     try:
#         if file_path.endswith('.csv'):
#             df = pd.read_csv(file_path)
#         elif file_path.endswith('.xlsx'):
#             df = pd.read_excel(file_path)
#         else:
#             raise ValueError("Unsupported file format. Only CSV and XLSX are allowed.")
        
#         logger.info(f"Data loaded successfully from {file_path}.")
#         return df
#     except Exception as e:
#         logger.error(f"Error loading data: {e}")
#         raise

def validate_data(df):
    """
    Validate the DataFrame using Great Expectations.
    Generates schema and statistics based on defined expectations,
    captures validation results, generates DataDocs (if possible), and saves results to a file.
    
    Parameters:
      df (pd.DataFrame): DataFrame to validate.
      
    Returns:
      dict: Validation results.
    """
    try:
        # Convert to a Great Expectations DataFrame
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        ge_df = ge.from_pandas(df)
        # logger.info(df["Product Name"].dtype)

        
        # Define expectations
        ge_df.expect_column_to_exist("Product Name")
        ge_df.expect_column_values_to_be_of_type("Product Name", "str")

        # ge_df.expect_column_to_exist("user_id")
        ge_df.expect_column_to_exist("Transaction ID")
 
        ge_df.expect_column_to_exist("Quantity")
        ge_df.expect_column_values_to_be_of_type("Quantity", "int")

        ge_df.expect_column_to_exist("Date")
        date_regex = (
            r"^(?:"
            r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2}(?:\.\d+)?){0,1}"
            r"|"
            r"\d{2}-\d{2}-\d{4}"
            r"|"
            r"\d{2}/\d{2}/\d{4}"
            r")$"
        )

        # 3. Set result_format="COMPLETE" to capture unexpected rows/values
        ge_df.expect_column_values_to_match_regex(
            "Date", date_regex, result_format="COMPLETE"
        )
        
        # Validate the dataset and capture results
        validation_results = ge_df.validate()

        validation_results_dict = validation_results.to_json_dict()
        
        # Attempt to generate DataDocs if a DataContext is available
        try:
            context = ge.data_context.DataContext()  # requires a GE config (great_expectations.yml)
            context.build_data_docs()
            logger.info("DataDocs generated successfully.")
        except Exception as doc_ex:
            logger.warning(f"DataDocs generation failed: {doc_ex}")
        
        # Save the validation results to a JSON file
        output_path = "validation_results.json"
        with open(output_path, "w") as f:
            json.dump(validation_results_dict, f, indent=2)
        logger.info(f"Validation results saved to {output_path}.")
        
        return validation_results_dict
    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        raise

def send_anomaly_alert(message):
    """
    Send an anomaly alert using email.
    
    Parameters:
      user_id (str/int): Identifier of the user.
      message (str): Alert message.
    """
    try:
        recipient_email = "patelmit640@gmail.com"
        send_email(recipient_email, subject="Anomaly Alert")
        logger.info(f"Data Validation Anomaly alert sent to user: {message}")
    except Exception as e:
        logger.error(f"Error sending anomaly alert: {e}")
        raise

def main(cloud: str = False):
    """
    Main function to run the entire workflow.
    This includes fetching the file from GCP, loading data, validating data,
    and sending alerts if any anomalies are detected.
    """
    try:
        # Retrieve bucket name dynamically; default to 'fully-processed-data'
        bucket_name = os.getenv("GCP_BUCKET_NAME", "full-raw-data")
        
        # Define the file name in the bucket; adjust as needed
        file_name = "temp_messy_transactions_20190103_20241231.xlsx"
        
        # Fetch file from GCP
        if cloud:
            df = load_bucket_data(bucket_name, file_name)
        else:
            df = load_data(file_name)
        
        
        # Validate data and generate schema/stats metadata
        validation_results = validate_data(df)
        
        # Example anomaly check: if the mean of 'quantity' exceeds a threshold.
        if df["Quantity"].mean() > 100:
            send_anomaly_alert(message="High demand detected!")
        
        logger.info("Workflow completed successfully.")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")

if __name__ == "__main__":
    main()
