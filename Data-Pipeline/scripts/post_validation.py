import os
import logging
import pandas as pd
from google.cloud import storage
import great_expectations as ge
import smtplib
from email.message import EmailMessage
import json


def setup_logging():
    """Set up logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def send_email(
    emailid,
    message,
    subject="Automated Email",
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    sender="svarunanusheel@gmail.com",
    username="svarunanusheel@gmail.com",
    password="Temp",
):
    """
    Sends an email to the given email address.

    Parameters:
      emailid (str): Recipient email address.
      message (str, pd.DataFrame, or list): Message content.
      subject (str): Email subject.
      smtp_server (str): SMTP server address.
      smtp_port (int): SMTP server port.
      sender (str): Sender's email address.
      username (str): SMTP username.
      password (str): SMTP password.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = emailid

    # Build the email content based on message type.
    if isinstance(message, str):
        msg.set_content(message)
    elif isinstance(message, pd.DataFrame):
        plain_text = message.to_string()
        html_text = message.to_html()
        msg.set_content(plain_text)
        msg.add_alternative(html_text, subtype="html")
    elif isinstance(message, list):
        text_parts = []
        html_parts = []
        for part in message:
            if isinstance(part, str):
                text_parts.append(part)
                html_parts.append(f"<p>{part}</p>")
            elif isinstance(part, pd.DataFrame):
                text_parts.append(part.to_string())
                html_parts.append(part.to_html())
            else:
                text_parts.append(str(part))
                html_parts.append(f"<p>{str(part)}</p>")
        combined_text = "\n".join(text_parts)
        combined_html = "".join(html_parts)
        msg.set_content(combined_text)
        msg.add_alternative(combined_html, subtype="html")
    else:
        msg.set_content(str(message))

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


def fetch_file_from_gcp(bucket_name, file_name, destination):
    """
    Fetch file from the specified GCP bucket and save it to the destination.

    Parameters:
      bucket_name (str): Name of the GCP bucket.
      file_name (str): Name/path of the file in the bucket.
      destination (str): Local destination path.
    """
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(destination)
        logger.info(f"File {file_name} downloaded from GCP bucket {bucket_name}.")
    except Exception as e:
        logger.error(f"Error fetching file from GCP: {e}")
        raise


def load_data(file_path):
    """
    Load data from a CSV or Excel file into a Pandas DataFrame.

    Parameters:
      file_path (str): Local path to the file.
    """
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and XLSX are allowed.")

        logger.info(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


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
        ge_df = ge.from_pandas(df)

        # Define expectations
        ge_df.expect_column_to_exist("product_id")
        ge_df.expect_column_to_exist("user_id")
        ge_df.expect_column_to_exist("transaction_date")
        ge_df.expect_column_to_exist("quantity")
        ge_df.expect_column_values_to_be_of_type("quantity", "int")

        # Validate the dataset and capture results
        validation_results = ge_df.validate()

        # Attempt to generate DataDocs if a DataContext is available
        try:
            context = (
                ge.data_context.DataContext()
            )  # requires a GE config (great_expectations.yml)
            context.build_data_docs()
            logger.info("DataDocs generated successfully.")
        except Exception as doc_ex:
            logger.warning(f"DataDocs generation failed: {doc_ex}")

        # Save the validation results to a JSON file
        output_path = "/tmp/validation_results.json"
        with open(output_path, "w") as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {output_path}.")

        return validation_results
    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        raise


def send_anomaly_alert(user_id, message):
    """
    Send an anomaly alert using email.

    Parameters:
      user_id (str/int): Identifier of the user.
      message (str): Alert message.
    """
    try:
        # Example: Send email alert to a predefined recipient (modify as needed)
        recipient_email = "alert@example.com"  # Replace with actual alert recipient
        email_message = f"Alert for user {user_id}: {message}"
        send_email(recipient_email, email_message, subject="Anomaly Alert")
        logger.info(f"Anomaly alert sent for user {user_id}: {message}")
    except Exception as e:
        logger.error(f"Error sending anomaly alert: {e}")
        raise


def main():
    """
    Main function to run the entire workflow.
    This includes fetching the file from GCP, loading data, validating data,
    and sending alerts if any anomalies are detected.
    """
    try:
        # Retrieve bucket name dynamically; default to 'fully-processed-data'
        bucket_name = os.getenv("GCP_BUCKET_NAME", "fully-processed-data")

        # Define the file name in the bucket; adjust as needed
        file_name = "transactions_20190103_20241231.xlsx"

        # Set local destination path
        destination = f"/tmp/{file_name}"

        # Fetch file from GCP
        fetch_file_from_gcp(bucket_name, file_name, destination)

        # Load data into DataFrame
        df = load_data(destination)

        # Validate data and generate schema/stats metadata
        validation_results = validate_data(df)

        # Example anomaly check: if the mean of 'quantity' exceeds a threshold.
        if df["quantity"].mean() > 100:
            send_anomaly_alert(
                user_id=df["user_id"].iloc[0], message="High demand detected!"
            )

        logger.info("Workflow completed successfully.")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")


if __name__ == "__main__":
    logger = setup_logging()
    main()
