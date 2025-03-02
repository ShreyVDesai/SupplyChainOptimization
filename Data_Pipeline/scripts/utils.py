import json
import os

import numpy as np
import pandas as pd
import polars as pl

try:
    from logger import logger
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger

import io
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()


# Set up GCP credentials path
def setup_gcp_credentials():
    """
    Sets up the GCP credentials by setting the GOOGLE_APPLICATION_CREDENTIALS environment variable
    to point to the correct location of the GCP key file.
    """
    # The GCP key is always in the mounted secret directory
    gcp_key_path = "/app/secret/gcp-key.json"

    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") != gcp_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key_path
        logger.info(f"Set GCP credentials path to: {gcp_key_path}")
    else:
        logger.info(f"Using existing GCP credentials from: {gcp_key_path}")


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

        file_extension = file_name.split(".")[-1].lower()
        if file_extension == "csv":
            try:
                df = pl.read_csv(io.BytesIO(blob_content))
                logger.info(
                    f"'{file_name}' from bucket '{bucket_name}' successfully read as CSV into DataFrame."
                )
            except Exception as e:
                logger.error(f"Error reading '{file_name}' as CSV: {e}")
                raise

        elif file_extension == "json":
            try:
                df = pd.read_json(io.BytesIO(blob_content))
                logger.info(
                    f"'{file_name}' from bucket '{bucket_name}' successfully read as JSON into DataFrame."
                )
            except Exception as e:
                logger.error(f"Error reading '{file_name}' as JSON: {e}")
                raise

        elif file_extension == "xlsx" or file_extension == "xls":
            try:
                df = pl.read_excel(io.BytesIO(blob_content))
                logger.info(
                    f"'{file_name}' from bucket '{bucket_name}' successfully read as Excel into DataFrame."
                )
            except Exception as e:
                logger.error(f"Error reading '{file_name}' as Excel: {e}")
                raise
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")

        if df.is_empty():
            error_msg = f"DataFrame loaded from bucket '{bucket_name}', file '{file_name}' is empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        return df

    except Exception as e:
        logger.error(
            f"Error occurred while loading data from bucket '{bucket_name}', file '{file_name}': {e}"
        )
        raise


def send_email(
    emailid,
    body,
    subject="Automated Email",
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    sender="talksick530@gmail.com",
    username="talksick530@gmail.com",
    password="celm dfaq qllh ymjv",
    attachment=None,
):
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

    # If an attachment is provided and it's a DataFrame, attach it as a CSV
    # file.
    if attachment is not None and isinstance(attachment, pd.DataFrame):
        csv_buffer = io.StringIO()
        attachment.to_csv(csv_buffer, index=False)
        # Encode the CSV content to bytes to avoid calling set_text_content.
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        msg.add_attachment(
            csv_bytes,
            maintype="text",
            subtype="csv",
            filename="anomalies.csv"
        )

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


def upload_to_gcs(
    df: pl.DataFrame, bucket_name: str, destination_blob_name: str
) -> None:
    """
    Uploads a DataFrame to Google Cloud Storage (GCS) in multiple formats (CSV, JSON, XLSX).

    Args:
        df (polars.DataFrame): The DataFrame to upload.
        bucket_name (str): The name of the GCS bucket where the file should be stored.
        destination_blob_name (str): The desired name for the file in GCS, which determines the file format.

    Returns:
        None

    Raises:
        Exception: If any other error occurs during the process.
    """
    setup_gcp_credentials()

    try:
        logger.info(
            "Starting upload to GCS. Bucket: %s, Blob: %s",
            bucket_name,
            destination_blob_name,
        )

        file_extension = destination_blob_name.split(".")[-1].lower()
        bucket = storage.Client().get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        if file_extension == "csv":
            csv_data = df.write_csv()
            blob.upload_from_string(csv_data, content_type="text/csv")
            logger.info("CSV data uploaded successfully")

        elif file_extension == "json":
            try:
                # First try standard write_json method
                json_data = df.write_json()
                blob.upload_from_string(
                    json_data, content_type="application/json"
                )
                logger.info("JSON data uploaded successfully using write_json")
            except Exception as e:
                logger.warning(
                    f"Standard JSON serialization failed: {e}, trying alternative approach"
                )

                # If that fails, try to convert to dict and then to JSON
                try:
                    # Helper function to convert numpy/pandas types to Python
                    # types
                    def convert_to_python_types(obj):
                        if isinstance(
                            obj,
                            (
                                np.int_,
                                np.intc,
                                np.intp,
                                np.int8,
                                np.int16,
                                np.int32,
                                np.int64,
                                np.uint8,
                                np.uint16,
                                np.uint32,
                                np.uint64,
                            ),
                        ):
                            return int(obj)
                        elif isinstance(
                            obj,
                            (np.float_, np.float16, np.float32, np.float64),
                        ):
                            return float(obj)
                        elif isinstance(obj, (np.bool_)):
                            return bool(obj)
                        elif isinstance(obj, (np.ndarray, )):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {
                                k: convert_to_python_types(v)
                                for k, v in obj.items()
                            }
                        elif isinstance(obj, list):
                            return [
                                convert_to_python_types(item) for item in obj
                            ]
                        else:
                            return obj

                    # If DataFrame has a single row and contains dictionaries,
                    # extract the first row
                    if df.shape[0] == 1 and df.shape[1] == 1:
                        cell_value = df[0, 0]
                        if isinstance(cell_value, dict):
                            data_dict = cell_value
                            logger.info(
                                "Found dictionary in single cell DataFrame"
                            )
                        else:
                            data_dict = df.to_pandas().to_dict(
                                orient="records"
                            )[0]
                    else:
                        data_dict = df.to_pandas().to_dict(orient="records")

                    # Convert numpy/pandas types to Python native types
                    python_typed_dict = convert_to_python_types(data_dict)

                    # Convert to JSON
                    json_data = json.dumps(python_typed_dict, indent=2)
                    blob.upload_from_string(
                        json_data, content_type="application/json"
                    )
                    logger.info(
                        "JSON data uploaded successfully using dict conversion"
                    )
                except Exception as e2:
                    logger.warning(
                        f"Dict conversion failed: {e2}, trying pandas direct conversion"
                    )
                    # Last resort: try pandas to_json
                    pd_df = df.to_pandas()
                    json_data = pd_df.to_json(orient="records")
                    blob.upload_from_string(
                        json_data, content_type="application/json"
                    )
                    logger.info(
                        "JSON data uploaded successfully using pandas conversion"
                    )

        # TODO: Add support for Excel files
        # elif file_extension == 'xlsx':
        #     excel_buffer = BytesIO()
        #     df.write_excel(excel_buffer, index=False, engine="openpyxl")
        #     excel_buffer.seek(0)
        #     blob.upload_from_file(excel_buffer, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        logger.info(
            "Upload successful to GCS. Blob name: %s", destination_blob_name
        )

    except Exception as e:
        logger.error("Error uploading DataFrame to GCS. Error: %s", e)
        raise


def list_bucket_blobs(bucket_name: str) -> list:
    """
    Lists all blobs in a Google Cloud Storage bucket.

    Parameters:
        bucket_name (str): The name of the GCS bucket.

    Returns:
        list: List of blob names in the bucket.

    Raises:
        Exception: If an error occurs during listing.
    """
    setup_gcp_credentials()
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()
        blob_names = [blob.name for blob in blobs]
        logger.info(f"Found {len(blob_names)} files in bucket '{bucket_name}'")
        return blob_names
    except Exception as e:
        logger.error(f"Error listing blobs in bucket '{bucket_name}': {e}")
        raise


def delete_blob_from_bucket(bucket_name: str, blob_name: str) -> bool:
    """
    Deletes a blob from a Google Cloud Storage bucket.

    Parameters:
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    setup_gcp_credentials()
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Blob {blob_name} deleted from bucket {bucket_name}")
        return True
    except Exception as e:
        logger.error(
            f"Error deleting blob {blob_name} from bucket {bucket_name}: {e}"
        )
        return False


def collect_validation_errors(
    df, missing_columns, error_indices, error_reasons
):
    """
    Collect validation errors and update error indices and reasons.

    Parameters:
      df: The DataFrame being validated.
      missing_columns: List of columns that are missing.
      error_indices: A set to store indices of rows with errors.
      error_reasons: A dictionary to store error reasons for each row.
    """
    if missing_columns:
        # If columns are missing, mark all rows as having errors
        for idx in range(len(df)):
            error_indices.add(idx)
            error_reasons[idx] = [
                f"Missing columns: {', '.join(missing_columns)}"
            ]


def send_anomaly_alert(
    df=None,
    subject="Data Anomaly Alert",
    message=None,
    recipient="patelmit640@gmail.com",
    anomalies=None,
):
    """
    Send an anomaly alert using email.

    Parameters:
      df (pd.DataFrame, optional): DataFrame containing anomalies to attach.
      subject (str): Email subject.
      message (str): Alert message.
      recipient (str): Email recipient.
      anomalies (Dict[str, pl.DataFrame], optional): Dictionary of anomaly DataFrames.

    Note:
      This function supports two modes of operation:
      1. Direct DataFrame mode: Pass a DataFrame to `df` parameter
      2. Anomalies dictionary mode: Pass a dictionary of DataFrames to `anomalies` parameter
    """
    try:
        if message is None:
            message = (
                "Hi,\n\n"
                "Anomalies have been detected in the dataset. "
                "Please see the attached CSV file for details.\n\n"
                "Thank you!"
            )

        # Handle the dictionary of anomalies case (from preprocessing.py)
        if anomalies is not None:
            anomaly_list = []
            for anomaly_type, anomaly_df in anomalies.items():
                if not anomaly_df.is_empty():
                    df_pd = anomaly_df.to_pandas()
                    df_pd["anomaly_type"] = anomaly_type
                    anomaly_list.append(df_pd)

            if anomaly_list:
                combined_df = pd.concat(anomaly_list, ignore_index=True)
                send_email(
                    emailid=recipient,
                    body=message,
                    subject=subject,
                    attachment=combined_df,
                )
                logger.info("Anomaly alert email sent.")
            else:
                logger.info("No anomalies detected; no alert email sent.")
        # Handle the direct DataFrame case (from post_validation.py)
        elif df is not None:
            send_email(recipient, subject=subject, body=message, attachment=df)
            logger.info(
                f"Data Validation Anomaly alert sent to user: {recipient}"
            )
        else:
            logger.info("No anomalies provided; no alert email sent.")

    except Exception as e:
        logger.error(f"Error sending anomaly alert: {e}")
        raise
