import polars as pl
import pandas as pd
import re
import json
from pandas_schema import Schema, Column
from pandas_schema.validation import (
    CustomElementValidation,
    InRangeValidation,
    MatchesPatternValidation,
)
from logger import logger
from utils import send_email, upload_to_gcs


# Define validation functions for custom checks
def check_string_type(value):
    return isinstance(value, str)


def check_int_type(value):
    try:
        # Check if it can be converted to int without decimal part
        if isinstance(value, (int, float)):
            return int(value) == float(value)
        return False
    except:
        return False


# Define the schema using pandas-schema
VALIDATION_SCHEMA = Schema(
    [
        # Product Name validation
        Column(
            "Product Name",
            [
                CustomElementValidation(lambda x: x is not None, "cannot be null"),
                CustomElementValidation(check_string_type, "must be a string"),
            ],
        ),
        # Total Quantity validation
        Column(
            "Total Quantity",
            [
                CustomElementValidation(lambda x: x is not None, "cannot be null"),
                CustomElementValidation(check_int_type, "must be an integer"),
                InRangeValidation(1, None, "must be greater than or equal to 1"),
            ],
        ),
        # Date validation
        Column(
            "Date",
            [
                CustomElementValidation(lambda x: x is not None, "cannot be null"),
                MatchesPatternValidation(
                    r"^(?:\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2}(?:\.\d+)?){0,1}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4})$",
                    "must be in valid date format",
                ),
            ],
        ),
    ]
)


def validate_data(df):
    """
    Validate the DataFrame using pandas-schema validation.
    Captures validation results, saves them to a JSON file, and if any anomalies
    (invalid rows) are detected, saves those records with anomaly reasons and sends an email alert.

    Parameters:
      df (pd.DataFrame or pl.DataFrame): DataFrame to validate.

    Returns:
      dict: JSON-serializable validation results.
    """
    try:
        # Convert Polars to Pandas if needed
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Initialize validation results structure
        validation_results = {"results": []}

        # Perform validation with pandas-schema
        errors = VALIDATION_SCHEMA.validate(df)

        # Track which rows have errors
        error_indices = set()
        error_reasons = {}

        # Process validation errors
        for error in errors:
            row_index = error.row
            column_name = error.column
            error_message = error.message

            error_indices.add(row_index)

            # Create a formatted error reason
            reason = f"{column_name} {error_message}"

            # Add to error_reasons dict
            if row_index in error_reasons:
                error_reasons[row_index].append(reason)
            else:
                error_reasons[row_index] = [reason]

            # Add to validation results
            result = {
                "success": False,
                "expectation_config": {
                    "expectation_type": f"expect_{error.message.split(' ')[0]}",
                    "kwargs": {"column": column_name},
                },
                "unexpected_index_list": [row_index],
                "unexpected_list": [str(df.iloc[row_index][column_name])],
            }
            validation_results["results"].append(result)

        # If no errors, add success entries for each column
        if not errors:
            for column in VALIDATION_SCHEMA.columns:
                column_name = column.name
                for validation in column.validations:
                    validation_type = type(validation).__name__

                    # Map validation type to expectation type
                    if isinstance(validation, InRangeValidation):
                        expectation_type = "expect_column_values_to_be_between"
                        kwargs = {
                            "column": column_name,
                            "min_value": validation.min_value,
                        }
                    elif isinstance(validation, MatchesPatternValidation):
                        expectation_type = "expect_column_values_to_match_regex"
                        kwargs = {"column": column_name}
                    else:
                        expectation_type = "expect_column_values_to_not_be_null"
                        kwargs = {"column": column_name}

                    validation_results["results"].append(
                        {
                            "success": True,
                            "expectation_config": {
                                "expectation_type": expectation_type,
                                "kwargs": kwargs,
                            },
                            "unexpected_index_list": [],
                            "unexpected_list": [],
                        }
                    )

        # Create anomalies DataFrame
        anomalies_df = pd.DataFrame()
        if error_indices:
            anomalies_df = df.iloc[list(error_indices)].copy()
            anomalies_df["anomaly_reason"] = anomalies_df.index.map(
                lambda idx: "; ".join(error_reasons.get(idx, []))
            )

        # Save the validation results to a JSON file
        output_path = "validation_results.json"
        with open(output_path, "w") as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {output_path}.")

        # If anomalies are found, send an email alert
        if not anomalies_df.empty:
            logger.warning(
                f"Anomalies detected! {len(anomalies_df)} rows failed validation."
            )
            send_anomaly_alert(
                anomalies_df,
                subject="Data Validation Anomalies",
                message="Data Validation Anomalies Found! Please find attached CSV file with anomaly reasons.",
            )
        else:
            logger.info("No anomalies detected. No email sent.")

        return validation_results

    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        raise


def generate_numeric_stats(
    df, filename, include_columns=["Total Quantity", "Unit Price"]
):
    """
    Generate summary statistics for all numeric columns in the DataFrame.

    Parameters:
      df (pd.DataFrame or pl.DataFrame): The input DataFrame.
      filename (str): Path to save the statistics as a JSON file.
      include_columns (list): List of column names to exclude from numeric stats.

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
                    "mean": float(group[col].mean()),
                    "std": float(group[col].std()),
                    "min": float(group[col].min()),
                    "max": float(group[col].max()),
                    "median": float(group[col].median()),
                    "25th_percentile": float(group[col].quantile(0.25)),
                    "75th_percentile": float(group[col].quantile(0.75)),
                    "skewness": float(group[col].skew()),
                    "kurtosis": float(group[col].kurt()),
                }
        grouped_stats[product] = stats

    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    try:
        json_df = pl.DataFrame([{"stats": grouped_stats}])
        upload_to_gcs(
            json_df, bucket_name="metadata_stats", destination_blob_name=filename
        )
        logger.info(f"Numeric statistics saved to {filename}.")
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


def post_validation(df: pl.DataFrame, file_name: str) -> bool:
    """
    Main function to run the entire workflow.
    This includes fetching the file from GCP, loading data, validating data,
    and sending alerts if any anomalies are detected.
    """
    try:
        # Validate data and generate schema/stats metadata
        validate_data(df)

        generate_numeric_stats(df, file_name)

        logger.info("Workflow completed successfully.")

        return True
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise
