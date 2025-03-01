import numpy as np
import polars as pl
import pandas as pd
# from scripts.logger import logger
from logger import logger
# from data_pipeline.scripts.logger import logger
import io
from google.cloud import storage
from dotenv import load_dotenv
from typing import Dict, Tuple
import smtplib
from email.message import EmailMessage
import os

load_dotenv()

# Set up GCP credentials path
def setup_gcp_credentials():
    """
    Sets up the GCP credentials by setting the GOOGLE_APPLICATION_CREDENTIALS environment variable
    to point to the correct location of the GCP key file.
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    gcp_key_path = os.path.join(project_root, "secret", "gcp-key.json")

    # Make sure the key exists
    if not os.path.exists(gcp_key_path):
        # Try an alternative path - current directory might be project root
        alt_path = os.path.join(
            os.path.dirname(script_dir), "..", "secret", "gcp-key.json"
        )
        if os.path.exists(alt_path):
            gcp_key_path = alt_path
        else:
            # Final fallback to direct path from container
            gcp_key_path = "secret/gcp-key.json"
            if not os.path.exists(gcp_key_path):
                logger.warning(
                    f"GCP key not found at {gcp_key_path}. Authentication may fail."
                )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key_path
    logger.info(f"Using GCP credentials from: {gcp_key_path}")


def load_data(file_path: str) -> pl.DataFrame:
    """
    Loads the dataset from the given file path.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        pl.DataFrame: Loaded DataFrame.
    """
    try:
        if file_path.lower().endswith(".xlsx"):
            df = pl.read_excel(file_path)

        logger.info(
            f"Data successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df

    except FileNotFoundError:
        logger.error(f"File Not Found: {file_path}")

    except Exception as e:
        logger.error(f"Fail to load data due to: {e}")
        raise e



def load_bucket_data(bucket_name: str, file_name: str) -> pl.DataFrame:
    """
    Loads data from a specified file in a Google Cloud Storage bucket and returns it as a Polars DataFrame.
    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        file_name (str): The name of the file within the bucket, including extension.

    Returns:
        pl.DataFrame: The content of the Excel file as a Polars DataFrame.

    Raises:
        Exception: If an error occurs while accessing the bucket or reading the file.
    """

    setup_gcp_credentials()

    try:
        bucket = storage.Client().get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob_content = blob.download_as_string()
        df = pl.read_excel(io.BytesIO(blob_content))
        logger.info(f"'{file_name}' from bucket '{bucket_name}' successfully read into DataFrame.")

        if df.is_empty():
            error_msg = f"DataFrame loaded from bucket '{bucket_name}', file '{file_name}' is empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        required_columns = [
            "Date",
            "Unit Price",
            "Quantity",
            "Transaction ID",
            "Store Location",
            "Product Name",
            "Producer ID"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"DataFrame loaded from bucket '{bucket_name}', file '{file_name}' " \
                         f"is missing required columns: {missing_columns}"
            
            logger.error(error_msg)
            raise ValueError(error_msg)

        return df

    except Exception as e:
        logger.error(f"Error occurred while loading data from bucket '{bucket_name}', file '{file_name}': {e}")
        raise



def send_email(emailid, body, subject="Automated Email", 
               smtp_server="smtp.gmail.com", smtp_port=587,
               sender="talksick530@gmail.com", username="talksick530@gmail.com", password="celm dfaq qllh ymjv",
               attachment=None):
    """
    Sends an email to the given email address with a message body.
    If an attachment (pandas DataFrame) is provided, it will be converted to CSV and attached.
    
    Parameters:
      emailid (str): Recipient email address.
      body (str): Email text content.
      subject (str): Subject of the email.
      smtp_server (str): SMTP server address.
      smtp_port (int): SMTP server port.
      sender (str): Sender's email address.
      username (str): Username for SMTP login.
      password (str): Password for SMTP login.
      attachment (pd.DataFrame, optional): If provided, attached as a CSV file.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = emailid
    msg.set_content(body)
    
    # If an attachment is provided and it's a DataFrame, attach it as a CSV file.
    if attachment is not None and isinstance(attachment, pd.DataFrame):
        csv_buffer = io.StringIO()
        attachment.to_csv(csv_buffer, index=False)
        # Encode the CSV content to bytes to avoid calling set_text_content.
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename="anomalies.csv")
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
        logger.info(f"Email sent successfully to: {emailid}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise