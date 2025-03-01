import os
import polars as pl
import pandas as pd
from google.cloud import storage
import great_expectations as ge
import smtplib
from email.message import EmailMessage
import json
from logger import logger
from utils import send_email, load_bucket_data, load_data, upload_to_gcs


COLUMN_EXPECTATIONS_AFTER_PREPROCESSING = [
    {
        "column": "Product Name",
        "expectations": [
            {
                "type": "expect_column_to_exist",
                "kwargs": {}
            },
            {
                "type": "expect_column_values_to_be_of_type",
                "kwargs": {"type_": "str"}
            }
        ]
    },
    {
        "column": "Total Quantity",
        "expectations": [
            {
                "type": "expect_column_to_exist",
                "kwargs": {}
            },
            {
                "type": "expect_column_values_to_be_of_type",
                "kwargs": {"type_": "int"}
            },
            {
                "type": "expect_column_values_to_be_between",
                "kwargs": {"min_value": 0}
            }
        ]
    },
    {
        "column": "Date",
        "expectations": [
            {
                "type": "expect_column_to_exist",
                "kwargs": {}
            },
            {
                "type": "expect_column_values_to_match_regex",
                "kwargs": {
                    "regex": (
                        r"^(?:"
                        r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2}(?:\.\d+)?){0,1}"
                        r"|"
                        r"\d{2}-\d{2}-\d{4}"
                        r"|"
                        r"\d{2}/\d{2}/\d{4}"
                        r")$"
                    )
                }
            }
        ]
    }
]



# def validate_data(df):
#     """
#     Validate the DataFrame using Great Expectations.
#     Generates schema and statistics based on defined expectations,
#     captures validation results, generates DataDocs (if possible), and saves results to a file.
    
#     Parameters:
#       df (pd.DataFrame): DataFrame to validate.
      
#     Returns:
#       dict: Validation results.
#     """
    # try:
    #     # Convert to a Great Expectations DataFrame
    #     if isinstance(df, pl.DataFrame):
    #         df = df.to_pandas()
    #     ge_df = ge.from_pandas(df)
    #     # logger.info(df["Product Name"].dtype)

        
    #     # Define expectations
    #     ge_df.expect_column_to_exist("Product Name")
    #     ge_df.expect_column_values_to_be_of_type("Product Name", "str")

    #     # ge_df.expect_column_to_exist("user_id")
    #     ge_df.expect_column_to_exist("Transaction ID")
 
    #     ge_df.expect_column_to_exist("Total Quantity")
    #     ge_df.expect_column_values_to_be_of_type("Total Quantity", "int")

    #     ge_df.expect_column_to_exist("Date")
    #     date_regex = (
    #         r"^(?:"
    #         r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2}(?:\.\d+)?){0,1}"
    #         r"|"
    #         r"\d{2}-\d{2}-\d{4}"
    #         r"|"
    #         r"\d{2}/\d{2}/\d{4}"
    #         r")$"
    #     )

    #     # 3. Set result_format="COMPLETE" to capture unexpected rows/values
    #     ge_df.expect_column_values_to_match_regex(
    #         "Date", date_regex, result_format="COMPLETE"
    #     )
        
    #     # Validate the dataset and capture results
    #     validation_results = ge_df.validate()

    #     validation_results_dict = validation_results.to_json_dict()
        
    #     # Attempt to generate DataDocs if a DataContext is available
    #     try:
    #         context = ge.data_context.DataContext()  # requires a GE config (great_expectations.yml)
    #         context.build_data_docs()
    #         logger.info("DataDocs generated successfully.")
    #     except Exception as doc_ex:
    #         logger.warning(f"DataDocs generation failed: {doc_ex}")
        
    #     # Save the validation results to a JSON file
    #     output_path = "validation_results.json"
    #     with open(output_path, "w") as f:
    #         json.dump(validation_results_dict, f, indent=2)
    #     logger.info(f"Validation results saved to {output_path}.")
        
    #     return validation_results_dict
    # except Exception as e:
    #     logger.error(f"Error in data validation: {e}")
    #     raise





    
def validate_data(df):
    """
    Validate the DataFrame using Great Expectations.
    Applies expectations dynamically from COLUMN_EXPECTATIONS,
    captures validation results, saves them to a JSON file, and if any anomalies
    (invalid rows) are detected, saves those records with anomaly reasons to a CSV
    and sends an email alert with the CSV attached.

    Parameters:
      df (pd.DataFrame or pl.DataFrame): DataFrame to validate.

    Returns:
      dict: JSON-serializable validation results.
    """
    try:
        # 1. Convert Polars to Pandas if needed
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # 2. Create a Great Expectations DataFrame
        ge_df = ge.from_pandas(df)


        # 3. Dynamically apply expectations from COLUMN_EXPECTATIONS
        for col_config in COLUMN_EXPECTATIONS_AFTER_PREPROCESSING:
            col_name = col_config["column"]
            for exp in col_config["expectations"]:
                exp_type = exp["type"]
                kwargs = exp["kwargs"].copy()
                kwargs["result_format"] = "COMPLETE"
                if "column" not in kwargs:
                    kwargs["column"] = col_name

                # Dynamically get the GE method and invoke it
                ge_method = getattr(ge_df, exp_type)
                ge_method(**kwargs)

        # 4. Validate the entire DataFrame and convert results to a JSON serializable dict
        validation_results = ge_df.validate()
        validation_results_dict = validation_results.to_json_dict()


        # 5. Save the validation results to a JSON file
        output_path = "validation_results.json"
        with open(output_path, "w") as f:
            json.dump(validation_results_dict, f, indent=2)
        logger.info(f"Validation results saved to {output_path}.")

        # 6. Gather all unexpected row indices and collect anomaly reasons
        anomaly_reasons = {}
        all_unexpected_indices = set()
        for result in validation_results.results:
            unexpected_indices = result.result.get("unexpected_index_list", [])
            exp_type = result.expectation_config.expectation_type
            col_name = result.expectation_config.kwargs.get("column")
            unexpected_vals = result.result.get("unexpected_list", [])
            if unexpected_vals:
                logger.warning(
                    f"Unexpected values found for '{col_name}' in {exp_type}: "
                    f"{unexpected_vals} at indices {unexpected_indices}"
                )
                for idx, val in zip(unexpected_indices, unexpected_vals):
                    reason = f"{col_name} fails {exp_type} (value: {val})"
                    if idx in anomaly_reasons:
                        anomaly_reasons[idx].append(reason)
                    else:
                        anomaly_reasons[idx] = [reason]
                    all_unexpected_indices.add(idx)

        # 7. Filter the original DataFrame to get rows that failed validation
        anomalies_df = df.iloc[list(all_unexpected_indices)].copy()
    
        if not anomalies_df.empty:
            anomalies_df["anomaly_reason"] = anomalies_df.index.map(
                lambda idx: "; ".join(anomaly_reasons.get(idx, []))
            )

        # 8. If anomalies are found, save them to CSV and send an email alert
        if not anomalies_df.empty:
            anomalies_csv = "data_validation_anomalies.csv"
            # anomalies_df.to_csv(anomalies_csv, index=False)
            logger.warning(
                f"Anomalies detected! {len(anomalies_df)} rows failed validation. "
            )
            send_anomaly_alert(
                anomalies_df,
                subject="Data Validation Anomalies",
                message="Data Validation Anomalies Found! Please find attached CSV file with anomaly reasons."
            )
        else:
            logger.info("No anomalies detected. No email sent.")

        return validation_results_dict

    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        raise


def generate_numeric_stats(df, output_path="numeric_stats.json", include_columns=["Total Quantity", "Unit Price"]):
    """
    Generate summary statistics for all numeric columns in the DataFrame.
    
    Parameters:
      df (pd.DataFrame or pl.DataFrame): The input DataFrame.
      output_path (str): Path to save the statistics as a JSON file.
      exclude_columns (list): List of column names to exclude from numeric stats.
      
    Returns:
      dict: A dictionary containing summary statistics (mean, std, min, max, median) for each numeric column.
    """
    # Convert Polars to Pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    grouped_stats = {}

    grouped = df.groupby("Product Name")
    for product, group in grouped:
        stats = {}
        for col in include_columns:
            if col in group.columns:
                stats[col] = {
                    "mean": group[col].mean(),
                    "std": group[col].std(),
                    "min": group[col].min(),
                    "max": group[col].max(),
                    "median": group[col].median(),
                    "25th_percentile": group[col].quantile(0.25),
                    "75th_percentile": group[col].quantile(0.75),
                    "skewness": group[col].skew(),
                    "kurtosis": group[col].kurt()
                }
        grouped_stats[product] = stats
    
    # Save statistics to a JSON file
    # with open(output_path, "w") as f:
    #     json.dump(grouped_stats, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x)
    try:
        json_df = pl.DataFrame([grouped_stats])
        upload_to_gcs(json_df, bucket_name="metadata_stats", destination_blob_name=output_path)
        logger.info(f"Numeric statistics saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error uploading statistics to GCS: {e}")
        raise
    
    return grouped_stats



def send_anomaly_alert(df, subject, message):
    """
    Send an anomaly alert using email.
    
    Parameters:
      user_id (str/int): Identifier of the user.
      message (str): Alert message.
    """
    try:
        recipient_email = "patelmit640@gmail.com"
        send_email(recipient_email, subject=subject, body=message, attachment=df)
        logger.info(f"Data Validation Anomaly alert sent to user: {message}")
    except Exception as e:
        logger.error(f"Error sending anomaly alert: {e}")
        raise

def main(cloud: str = True):
    """
    Main function to run the entire workflow.
    This includes fetching the file from GCP, loading data, validating data,
    and sending alerts if any anomalies are detected.
    """
    try:
        # Retrieve bucket name dynamically; default to 'fully-processed-data'
        bucket_name = os.getenv("GCP_BUCKET_NAME", "fully-processed-data")
        
        # Define the file name in the bucket; adjust as needed
        file_name = "processed_messy_transactions_20190103_20241231_20250301_053630.csv"
        
        # Fetch file from GCP
        if cloud:
            df = load_bucket_data(bucket_name, file_name)
        else:
            df = load_data(file_name)
        
        # Validate data and generate schema/stats metadata
        validation_results = validate_data(df)

        stats = generate_numeric_stats(df)
        
        logger.info("Workflow completed successfully.")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")

if __name__ == "__main__":
    main()
