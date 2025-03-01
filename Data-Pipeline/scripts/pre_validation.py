import os
import polars as pl
import pandas as pd
import great_expectations as ge
import json
import smtplib
from email.message import EmailMessage
from logger import logger
from utils import send_email, load_bucket_data, load_data

def validate_data(df):
    """
    Validate the DataFrame to ensure data integrity.
    
    Returns:
      bool: True if all checks pass, False otherwise.
    """
    try:
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        ge_df = ge.from_pandas(df)
        
        validation_errors = []

        # Column existence checks
        required_columns = [
            "Date", "Unit Price", "Transaction ID", "Quantity",
            "Producer ID", "Store Location", "Product Name"
        ]
        for col in required_columns:
            if col not in df.columns:
                validation_errors.append(f"Missing column: {col}")

        # Data type validations
        type_checks = {
            "Unit Price": ["float", "int"],
            "Transaction ID": ["int", "str"],
            "Quantity": ["int"],
            "Producer ID": ["int", "str"],
            "Store Location": ["str"],
            "Product Name": ["str"]
        }

        for col, valid_types in type_checks.items():
            if col in df.columns:
                actual_type = df[col].dtype.name
                if not any(actual_type.startswith(t) for t in valid_types):
                    validation_errors.append(f"Invalid type for {col}: Expected {valid_types}, found {actual_type}")

        # Date validation
        if "Date" in df.columns:
            date_regex = (
                r"^(?:"  # Matches formats like YYYY-MM-DD or MM/DD/YYYY
                r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2}(?:\.\d+)?)?"  # YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
                r"|"
                r"\d{2}-\d{2}-\d{4}"  # DD-MM-YYYY
                r"|"
                r"\d{2}/\d{2}/\d{4}"  # MM/DD/YYYY
                r")$"
            )
            date_check = ge_df.expect_column_values_to_match_regex("Date", date_regex, result_format="COMPLETE")
            if not date_check["success"]:
                validation_errors.append("Invalid date format detected.")

        # Quantity should be positive
        if "Quantity" in df.columns and (df["Quantity"] < 0).any():
            validation_errors.append("Negative values found in Quantity.")

        # Unit Price should be positive
        if "Unit Price" in df.columns and (df["Unit Price"] < 0).any():
            validation_errors.append("Negative values found in Unit Price.")

        # If any errors exist, send an email and return False
        if validation_errors:
            error_message = "\n".join(validation_errors)
            send_email(
                "patelmit640@gmail.com",
                subject="Data Validation Failed",
                body=f"Data validation failed with the following issues:\n\n{error_message}"
            )
            logger.error(f"Data validation failed:\n{error_message}")
            return False

        logger.info("Data validation passed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        return False

def main(cloud: bool = False):
    """
    Main function to run the validation workflow.
    """
    try:
        bucket_name = os.getenv("GCP_BUCKET_NAME", "full-raw-data")
        file_name = "messy_transactions_20190103_20241231.xlsx"

        df = load_bucket_data(bucket_name, file_name) if cloud else load_data(file_name)
        
        if not validate_data(df):
            logger.error("Validation failed. Exiting process.")
            return
        
        logger.info("Workflow completed successfully.")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")

if __name__ == "__main__":
    main()
