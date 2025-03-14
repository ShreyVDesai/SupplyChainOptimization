import os
import pickle
import logging
import math
import numpy as np
import pandas as pd
import mysql.connector

from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)

# -----------------------
# 1. HELPER FUNCTIONS
# -----------------------

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
    # gcp_key_path = "/app/secret/gcp-key.json" use when dockerizing
    gcp_key_path = "../../secret/gcp-key.json"

    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") != gcp_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key_path
        logger.info(f"Set GCP credentials path to: {gcp_key_path}")
    else:
        logger.info(f"Using existing GCP credentials from: {gcp_key_path}")


def load_model(bucket_name: str, file_name: str):
    """
    Loads a pickle file (typically a model) from a GCP bucket and returns the loaded object.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        file_name (str): The name of the pickle file in the bucket.

    Returns:
        The Python object loaded from the pickle file.

    Raises:
        Exception: If an error occurs during the download or unpickling process.
    """
    setup_gcp_credentials()

    try:
        bucket = storage.Client().get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob_content = blob.download_as_string()

        file_extension = file_name.split('.')[-1].lower()
        if file_extension not in ['pkl', 'pickle']:
            logger.error(f"Unsupported file type for pickle: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")

        model = pickle.load(io.BytesIO(blob_content))
        logger.info(f"'{file_name}' from bucket '{bucket_name}' successfully loaded as pickle.")
        return model

    except Exception as e:
        logger.error(
            f"Error occurred while loading pickle file from bucket '{bucket_name}', file '{file_name}': {e}"
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
            csv_bytes, maintype="text", subtype="csv", filename="anomalies.csv"
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

# def load_model(pickle_path: str):
#  
#     """Loads a pickle file and returns the model object. Works in local dev"""
#     with open(pickle_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

def get_latest_data_from_mysql(host, user, password, database, query):
    """Connects to MySQL, runs a query, returns a DataFrame of the results."""
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))

def compute_mape(y_true, y_pred):
    """
    Computes Mean Absolute Percentage Error.
    y_true: array-like of actual values
    y_pred: array-like of predicted values
    Returns MAPE in percentage. 
    Ignores zero or near-zero actual values to avoid division-by-zero errors.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def ks_test_drift(ref_data, new_data, alpha=0.05):
    """
    Performs the Kolmogorov–Smirnov test to detect distribution drift.
    Returns True if drift is detected, else False.
    
    alpha (float): significance level. If p-value < alpha => drift.
    """
    stat, p_value = ks_2samp(ref_data, new_data)
    return p_value < alpha  # True => drift

# -----------------------
# 2. DRIFT DETECTION
# -----------------------
def check_data_drift(ref_df, new_df, numeric_cols=None, alpha=0.05):
    """
    Checks data drift for numeric columns by running K-S test on each column.
    ref_df: reference dataset (training or older stable data)
    new_df: new dataset
    numeric_cols: list of columns to test. If None, will auto-detect numeric columns.
    alpha: significance level for K-S test
    Returns: (bool, list_of_drifts)
        - bool: True if at least one column drifted significantly
        - list_of_drifts: list of columns that show drift
    """
    if numeric_cols is None:
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    
    drift_detected_cols = []
    for col in numeric_cols:
        if col not in new_df.columns:
            continue  # skip if col missing in new_df
        ref_values = ref_df[col].dropna()
        new_values = new_df[col].dropna()
        
        if len(ref_values) < 2 or len(new_values) < 2:
            continue  # skip if not enough data for test
        
        drift_found = ks_test_drift(ref_values, new_values, alpha)
        if drift_found:
            drift_detected_cols.append(col)
    
    return (len(drift_detected_cols) > 0, drift_detected_cols)

def check_concept_drift(ref_errors, new_errors, alpha=0.05):
    """
    Performs concept drift detection by comparing the distribution of 
    residuals (errors) between reference data and new data.
    If the error distributions differ significantly (K-S test), we assume concept drift.
    
    alpha (float): significance level. If p-value < alpha => concept drift.
    Returns True if concept drift is detected, else False.
    """
    if len(ref_errors) < 2 or len(new_errors) < 2:
        return False
    return ks_test_drift(ref_errors, new_errors, alpha)

def get_latest_data_from_cloud_sql_tcp(host, port, user, password, database, query):
    """
    Connects to a Google Cloud SQL instance using TCP (public IP or Cloud SQL Proxy)
    and returns query results as a DataFrame.
    
    Args:
        host (str): The Cloud SQL instance IP address or localhost (if using Cloud SQL Proxy).
        port (int): The port number (typically 3306 for MySQL).
        user (str): Database username.
        password (str): Database password.
        database (str): Database name.
        query (str): SQL query to execute.
        
    Returns:
        pd.DataFrame: Query results.
    """
    conn = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -----------------------
# 3. MAIN SCRIPT
# -----------------------
def main():
    # 1. Load configuration from environment variables or config file
    instance_conn_str = "primordial-veld-450618-n4:us-central1:mlops-sql"
    host = os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    model_pickle_path = os.getenv("MODEL_PICKLE_PATH", "model.pkl")
    
    # Performance thresholds (example)
    rmse_threshold = float(os.getenv("RMSE_THRESHOLD", 10.0))
    mape_threshold = float(os.getenv("MAPE_THRESHOLD", 50.0))     # in percentage
    alpha_drift = float(os.getenv("ALPHA_DRIFT", 0.05))          # significance level for K-S tests

    # # 2. Fetch new data from MySQL (past 7 days)
    query_new_data = """
        SELECT 
            date, product_name, total_quantity
        FROM Sales 
        WHERE date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        ORDER BY date;
    """
    # new_df = get_latest_data_from_mysql(host, user, password, database, query_new_data)
    
    # 3. [Optional] Fetch reference data from MySQL (e.g., for drift detection)
    #    For example, reference might be the training dataset or a stable historical window.
    # query_ref_data = """
    #     SELECT 
    #         date, feature1, feature2, actual_value
    #     FROM your_table 
    #     WHERE date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    #       AND date < CURDATE()
    #     ORDER BY date;
    # """
    ref_df = get_latest_data_from_cloud_sql(
    instance_connection_string=instance_conn_str,
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
    query=query_ref_data
)

    # 4. Load current model
    logging.info(f"Loading model from {model_pickle_path}...")
    current_model = load_model(,'model.pkl')

    # 5. Preprocess new data (minimal example)
    new_features = new_df[["feature1", "feature2"]]  # adapt as needed
    y_true_new = new_df["actual_value"]
    
    # 6. Generate predictions on new data
    y_pred_new = current_model.predict(new_features)

    # 7. Compute Metrics
    rmse_new = compute_rmse(y_true_new, y_pred_new)
    mape_new = compute_mape(y_true_new, y_pred_new)
    logging.info(f"Weekly Validation Results:")
    logging.info(f"  - RMSE: {rmse_new:.4f}")
    logging.info(f"  - MAPE: {mape_new:.2f}%")

    # 8. Check if metrics exceed thresholds
    degraded = False
    if rmse_new > rmse_threshold:
        logging.warning(f"RMSE ({rmse_new:.4f}) exceeds threshold ({rmse_threshold}).")
        degraded = True
    if mape_new > mape_threshold:
        logging.warning(f"MAPE ({mape_new:.2f}%) exceeds threshold ({mape_threshold}%).")
        degraded = True

    # 9. Data Drift Detection
    #    Compare columns in reference data vs new data
    data_drift_detected, drifted_cols = check_data_drift(
        ref_df, new_df, 
        numeric_cols=["feature1", "feature2", "actual_value"],  # or None to auto-detect
        alpha=alpha_drift
    )

    if data_drift_detected:
        logging.warning(f"Data drift detected in columns: {drifted_cols}")

    # 10. Concept Drift Detection
    #     Compare residual distributions of ref vs new
    if not ref_df.empty:
        # Predict on reference data to get errors
        ref_features = ref_df[["feature1", "feature2"]]
        y_true_ref = ref_df["actual_value"]
        y_pred_ref = current_model.predict(ref_features)

        ref_errors = y_true_ref - y_pred_ref
        new_errors = y_true_new - y_pred_new
        
        concept_drift = check_concept_drift(ref_errors, new_errors, alpha=alpha_drift)
        if concept_drift:
            logging.warning("Concept drift detected based on error distribution shift.")
    else:
        logging.info("Reference data is empty; cannot perform concept drift check.")

    # 11. Decide retraining
    #     If metric thresholds are breached or drift is detected, trigger retraining
    if degraded or data_drift_detected or (not ref_df.empty and concept_drift):
        logging.warning("Triggering retraining due to performance or drift issues...")
        trigger_retraining()  # implement your own function or logic
    else:
        logging.info("Model performance and data distribution are within expected ranges.")


def trigger_retraining():
    """
    You can implement any mechanism here to trigger your training pipeline:
      - Call a Cloud Function endpoint.
      - Publish a Pub/Sub message that triggers the training job.
      - Use the Vertex AI Python SDK to start a training pipeline or custom job.
    """
    logging.info("Retraining triggered... ")
    def trigger_retraining_vertex_pipeline():
        aiplatform.init(project="your_project_id", location="us-central1")
    # If you have a pipeline already deployed:
    pipeline_job = aiplatform.PipelineJob(
        display_name="retraining-pipeline",
        template_path="gs://path-to-your-pipeline-spec.json",
        parameter_values={}
    )
    pipeline_job.run()

if __name__ == "__main__":
    main()
