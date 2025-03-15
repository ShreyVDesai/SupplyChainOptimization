import os
import pickle

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import math
import numpy as np
import pandas as pd
# # from google.cloud.sql.connector import Connector
# import sqlalchemy
from utils import get_latest_data_from_cloud_sql

from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp



# -----------------------
# 1. HELPER FUNCTIONS
# -----------------------

# try:
#     from logger import logger
# except ImportError:  # For testing purposes
#     # from ML_Models.scripts.logger import logger
#     raise

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
    gcp_key_path = "secret/gcp-key.json"

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

# def get_latest_data_from_mysql(host, user, password, database, query):
#     """Connects to MySQL, runs a query, returns a DataFrame of the results."""
#     conn = mysql.connector.connect(
#         host=host,
#         user=user,
#         password=password,
#         database=database
#     )
#     df = pd.read_sql(query, conn)
#     print(df.head())
#     conn.close()
#     return df


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
        numeric_cols = ref_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

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



# def get_latest_data_from_cloud_sql(host, user, password, database, query, port ='3306'):
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
    connector = Connector()
    def getconn():
        conn = connector.connect(
            "primordial-veld-450618-n4:us-central1:mlops-sql", # Cloud SQL instance connection name
            "pymysql",                    # Database driver
            user=user,                  # Database user
            password=password,          # Database password
            db=database,   
        )
        return conn
    pool = sqlalchemy.create_engine(
    "mysql+pymysql://", # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
    creator=getconn,
    )
    with pool.connect() as db_conn:
        result = db_conn.execute(sqlalchemy.text(query))
        print(result.scalar())
    df = pd.read_sql(query, pool)
    print(df.head())
    connector.close()
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
    rmse_threshold = float(os.getenv("RMSE_THRESHOLD", 20.0))
    mape_threshold = float(os.getenv("MAPE_THRESHOLD", 50.0))     # in percentage
    alpha_drift = float(os.getenv("ALPHA_DRIFT", 0.05))          # significance level for K-S tests

    # # 2. Fetch new data from MySQL (past 7 days)
    
    query_new_data = """
        SELECT 
            date, product_name, total_quantity
        FROM SALES
        WHERE date BETWEEN 
            (SELECT DATE_SUB(MAX(date), INTERVAL 13 DAY) FROM SALES)
        AND (SELECT DATE_SUB(MAX(date), INTERVAL 7 DAY) FROM SALES)
        ORDER BY date;
    """
    # query_new_data = "SELECT * FROM SALES"
    new_df = get_latest_data_from_cloud_sql(
    # instance_connection_string=instance_conn_str,
    
    query=query_new_data
)
    
    # 3. [Optional] Fetch reference data from MySQL (e.g., for drift detection)
    #    For example, reference might be the training dataset or a stable historical window.
    query_ref_data = """
        SELECT 
            date, product_name, total_quantity
        FROM SALES
        WHERE date BETWEEN 
            (SELECT DATE_SUB(MAX(date), INTERVAL 6 DAY) FROM SALES)
        AND (SELECT MAX(date) FROM SALES)
        ORDER BY date;
    """
    ref_df = get_latest_data_from_cloud_sql(
    query=query_ref_data
)

    # Convert the 'date' column to datetime 
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    ref_df['date'] = pd.to_datetime(ref_df['date'])
    
    # 4. Load current model
    logging.info(f"Loading model from {model_pickle_path}...")
    current_model = load_model('trained-model-1', 'model.pkl')
    
    # 5. Group data by product and calculate product-level predictions and metrics
    products = new_df['product_name'].unique()

    rmse_list = []
    data_drift_list = []
    concept_drift_list = []
    
    for product in products:
        # Filter data for the product
        prod_new = new_df[new_df['product_name'] == product].copy()
        prod_ref = ref_df[ref_df['product_name'] == product].copy() 
        
        # Set 'date' as index within the product group and sort
        prod_new.set_index('date', inplace=True)
        prod_new.sort_index(inplace=True)
        
        prod_ref.set_index('date', inplace=True)
        prod_ref.sort_index(inplace=True)
        
        # Define prediction window for this product
        start_date = prod_new.index.min()
        end_date = prod_new.index.max()
        
        # Generate predictions using the current model.
        # (Modify this if your model requires product-specific adjustments.)
        y_pred_prod = current_model.predict(start=start_date, end=end_date)
        y_true_prod = prod_new["total_quantity"]
        
        # Compute product-level RMSE
        rmse_prod = compute_rmse(y_true_prod, y_pred_prod)
        rmse_list.append(rmse_prod)
        logging.info(f"RMSE for product {product}: {rmse_prod:.4f}")
        
        # Data drift detection (for total_quantity)
        if not prod_ref.empty:
            drift_detected, drift_cols = check_data_drift(prod_ref, prod_new, numeric_cols=['total_quantity'], alpha=alpha_drift)
            data_drift_list.append((product, drift_detected, drift_cols))
            
            # Concept drift detection using error distributions
            start_ref = prod_ref.index.min()
            end_ref = prod_ref.index.max()
            y_pred_ref = current_model.predict(start=start_ref, end=end_ref)
            y_true_ref = prod_ref["total_quantity"]
            
            ref_errors = y_true_ref - y_pred_ref
            new_errors = y_true_prod - y_pred_prod
            concept_drift_detected = check_concept_drift(ref_errors, new_errors, alpha=alpha_drift)
            concept_drift_list.append((product, concept_drift_detected))
        else:
            data_drift_list.append((product, False, []))
            concept_drift_list.append((product, False))
    
    # 6. Average product-level RMSE for overall RMSE
    if rmse_list:
        final_rmse = sum(rmse_list) / len(rmse_list)
        logging.info(f"Final averaged RMSE across products: {final_rmse:.4f}")
    else:
        logging.info("No RMSE values computed; check product grouping.")
    
    # 7. Log drift detection results per product
    # for product, drift_detected, drift_cols in data_drift_list:
    #     if drift_detected:
    #         logging.warning(f"Data drift detected for product {product} in columns: {drift_cols}")
    #     else:
    #         logging.info(f"No data drift detected for product {product}.")
    # for product, cd in concept_drift_list:
    #     if cd:
    #         logging.warning(f"Concept drift detected for product {product}.")
    #     else:
    #         logging.info(f"No concept drift detected for product {product}.")
    
    # 8. Decide on retraining based on thresholds or drift
    if any(rmse > rmse_threshold for rmse in rmse_list) or \
       any(d[1] for d in data_drift_list) or \
       any(cd[1] for cd in concept_drift_list):
        logging.warning("Triggering retraining due to performance or drift issues...")
        # trigger_retraining()  # Implement your retraining logic here
    else:
        logging.info(
            "Model performance and data distribution are within expected ranges."
        )

if __name__ == "__main__":
    main()