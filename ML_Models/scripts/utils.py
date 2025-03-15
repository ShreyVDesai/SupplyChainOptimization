import pandas as pd
from google.cloud.sql.connector import Connector
import sqlalchemy
import os
import math
from sklearn.metrics import mean_squared_error
import io
import smtplib
from email.message import EmailMessage
import polars as pl
from Data_Pipeline.scripts.logger import logger


def extracting_time_series_and_lagged_features_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, compute additional time-series features:
      - day_of_week, is_weekend, day_of_month, day_of_year, month, week_of_year
      - lag_1, lag_7, and rolling_mean_7 of 'Total Quantity'
    """
    # If the DataFrame is empty, return an empty DataFrame with the expected columns.
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date", "Product Name", "Total Quantity", "day_of_week",
                "is_weekend", "day_of_month", "day_of_year", "month",
                "week_of_year", "lag_1", "lag_7", "rolling_mean_7"
            ]
        )

    # Ensure the 'Date' column is in datetime format
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Create date-based features
        df["day_of_week"] = df["Date"].dt.dayofweek  # Monday=0, Sunday=6
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_month"] = df["Date"].dt.day
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["month"] = df["Date"].dt.month
        # Using isocalendar to get week of year (as integer)
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    else:
        return df

    # Check if 'Total Quantity' exists and then compute lag and rolling features
    if "Total Quantity" in df.columns:
        # Sort by product and date
        df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
        
        # Compute lag features by grouping by 'Product Name'
        df["lag_1"] = df.groupby("Product Name")["Total Quantity"].shift(1)
        df["lag_7"] = df.groupby("Product Name")["Total Quantity"].shift(7)
        
        # Compute a rolling mean of the previous 7 days.
        # Note: We shift by 1 to ensure that the rolling window uses data strictly before the current row.
        df["rolling_mean_7"] = df.groupby("Product Name")["Total Quantity"].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
        )
    else:
        raise KeyError("Column 'Total Quantity' not found in DataFrame.")
    
    return df


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

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))

def get_latest_data_from_cloud_sql(query, port ='3306'):
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
    host = os.getenv("MYSQL_HOST")
    user=os.getenv("MYSQL_USER")
    password=os.getenv("MYSQL_PASSWORD")
    database=os.getenv("MYSQL_DATABASE")
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
