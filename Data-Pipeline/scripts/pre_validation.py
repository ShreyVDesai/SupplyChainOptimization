import pandas as pd
from sendMail import send_email
from dataPreprocessing import setup_gcp_credentials,load_bucket_data

def validate_file():
    """
    Validates a CSV file from a GCP bucket and sends an email with issues if found.
    
    Returns:
        bool: True if file is valid, False otherwise.
    """
    bucket_name = "full-raw-data"
    file_name = "messy_transactions_20190103_20241231.xlsx"
    emailid = "talksick530@gmail.com"
    data = load_bucket_data(bucket_name,file_name)
    
    issues = []
    
    # Check for correct headers
    required_headers = {"Date", "Unit Price", "Quantity", "Product Name"}
    if not required_headers.issubset(data.columns):
        issues.append("Missing required headers in the file.")
    
    # Validate Date column format
    try:
        pd.to_datetime(data['Date'])
    except Exception:
        issues.append("Invalid date format in 'Date' column.")
    
    # Validate numerical columns (e.g., 'Unit Price', 'Quantity')
    if (data['Unit Price'] < 0).sum() > 0:
        issues.append("Negative values found in 'Unit Price'.")
    if (data['Quantity'] <= 0).sum() > 0:
        issues.append("Invalid values found in 'Quantity'.")
    
    # Check for special characters in 'Product Name'
    if data['Product Name'].str.contains(r'[^a-zA-Z0-9 ]').sum() > 0:
        issues.append("Special characters found in 'Product Name'.")
    
    if issues:
        send_email(emailid, "\n".join(issues), "Data Validation Issues")
        return False
    
    return True

if __name__ == "__main__":
    validate_file()
