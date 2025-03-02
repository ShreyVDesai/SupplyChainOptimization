import pandas as pd
import polars as pl

try:
    from logger import logger
    from utils import (
        collect_validation_errors,
        send_anomaly_alert,
        upload_to_gcs,
    )
except ImportError:  # For testing purposes
    from Data_Pipeline.scripts.logger import logger
    from Data_Pipeline.scripts.utils import (
        collect_validation_errors,
        send_anomaly_alert,
        upload_to_gcs,
    )

# Post-validation expected columns
POST_VALIDATION_COLUMNS = ["Product Name", "Total Quantity", "Date"]


def check_column_types(df, error_indices, error_reasons):
    """
    Check column types and data validity using pandas functionality only.

    Parameters:
        df (pd.DataFrame): DataFrame to validate
        error_indices (set): Set to track indices of rows with errors
        error_reasons (dict): Dictionary to track error reasons by row index
    """
    # Check Product Name (should be string type)
    if "Product Name" in df.columns:
        invalid_product_mask = ~df["Product Name"].apply(
            lambda x: isinstance(x, str)
        )
        for idx in df[invalid_product_mask].index:
            error_indices.add(idx)
            reason = "Product Name must be a string"
            if idx in error_reasons:
                error_reasons[idx].append(reason)
            else:
                error_reasons[idx] = [reason]

    # Check Total Quantity (should be numeric and > 0)
    if "Total Quantity" in df.columns:
        # Check if numeric
        try:
            # Convert to numeric if possible, errors='coerce' will set invalid
            # values to NaN
            quantity_series = pd.to_numeric(
                df["Total Quantity"], errors="coerce"
            )

            # Check for NaN values (conversion failures)
            nan_mask = quantity_series.isna()
            for idx in df[nan_mask].index:
                error_indices.add(idx)
                reason = "Total Quantity must be numeric"
                if idx in error_reasons:
                    error_reasons[idx].append(reason)
                else:
                    error_reasons[idx] = [reason]

            # Check for values <= 0 (only for valid numeric values)
            invalid_quantity_mask = (quantity_series <= 0) & ~nan_mask
            for idx in df[invalid_quantity_mask].index:
                error_indices.add(idx)
                reason = "Total Quantity must be greater than 0"
                if idx in error_reasons:
                    error_reasons[idx].append(reason)
                else:
                    error_reasons[idx] = [reason]
        except Exception as e:
            logger.error(f"Error validating Total Quantity: {e}")

    # Check Date (should be convertible to datetime)
    if "Date" in df.columns:
        for idx in df.index:
            date_val = df.loc[idx, "Date"]
            try:
                # Try to convert to datetime
                if pd.isna(date_val):
                    error_indices.add(idx)
                    reason = "Date cannot be null"
                    if idx in error_reasons:
                        error_reasons[idx].append(reason)
                    else:
                        error_reasons[idx] = [reason]
                else:
                    # Check if it matches expected date format using string
                    # operations
                    if isinstance(date_val, str):
                        # Check various date formats using string manipulation
                        date_formats_valid = any(
                            [
                                # YYYY-MM-DD
                                len(date_val) >= 10
                                and date_val[4] == "-"
                                and date_val[7] == "-",
                                # MM-DD-YYYY or DD-MM-YYYY
                                len(date_val) >= 10
                                and date_val[2] == "-"
                                and date_val[5] == "-",
                                # MM/DD/YYYY or DD/MM/YYYY
                                len(date_val) >= 10
                                and date_val[2] == "/"
                                and date_val[5] == "/",
                            ]
                        )

                        if not date_formats_valid:
                            error_indices.add(idx)
                            reason = "Date must be in a valid date format"
                            if idx in error_reasons:
                                error_reasons[idx].append(reason)
                            else:
                                error_reasons[idx] = [reason]
            except Exception as e:
                error_indices.add(idx)
                reason = f"Invalid date format: {e}"
                if idx in error_reasons:
                    error_reasons[idx].append(reason)
                else:
                    error_reasons[idx] = [reason]


def validate_data(df):
    """
    Validate the DataFrame by checking if it contains all required columns
    and has at least one row of data. Also performs type checking on columns.
    Captures validation results, saves them to a JSON file, and if any anomalies
    are detected, saves those records with anomaly reasons and sends an email alert.

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
        validation_results = {
            "results": [],
            "has_errors": False,
            "missing_columns": [],
            "error_count": 0,
        }
        error_indices = set()
        error_reasons = {}

        # Check if all required columns are present
        missing_columns = [
            col for col in POST_VALIDATION_COLUMNS if col not in df.columns
        ]

        # Check if DataFrame is empty
        if len(df) == 0:
            error_message = "DataFrame is empty, no rows found"
            for idx in range(len(df)):
                error_indices.add(idx)
                error_reasons[idx] = [error_message]

        # If columns are missing, collect the errors
        if missing_columns:
            collect_validation_errors(
                df, missing_columns, error_indices, error_reasons
            )
        else:
            # Only perform type checking if all required columns are present
            check_column_types(df, error_indices, error_reasons)

        # Create anomalies DataFrame
        anomalies_df = pd.DataFrame()
        if error_indices:
            anomalies_df = (
                df.iloc[list(error_indices)].copy()
                if not df.empty
                else pd.DataFrame()
            )
            # Only add anomaly_reason if there are actual errors
            if not anomalies_df.empty:
                anomalies_df["anomaly_reason"] = anomalies_df.index.map(
                    lambda idx: "; ".join(error_reasons.get(idx, []))
                )

        # If anomalies are found, send an email alert
        if not anomalies_df.empty or missing_columns:
            error_message = ""
            if missing_columns:
                error_message += (
                    f"Missing columns: {', '.join(missing_columns)}. "
                )
            if not anomalies_df.empty:
                error_message += f"{len(anomalies_df)} rows failed validation."

            logger.warning(f"Anomalies detected! {error_message}")
            send_anomaly_alert(
                df=anomalies_df,
                subject="Data Validation Anomalies",
                message=f"Data Validation Anomalies Found! {error_message} Please find attached CSV file with anomaly details.",
            )
        else:
            logger.info("No anomalies detected. No email sent.")

        validation_results["has_errors"] = len(error_indices) > 0
        validation_results["missing_columns"] = missing_columns
        validation_results["error_count"] = len(error_indices)

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
            json_df,
            bucket_name="metadata_stats",
            destination_blob_name=filename,
        )
        logger.info(f"Numeric statistics saved to {filename}.")
    except Exception as e:
        logger.error(f"Error uploading statistics to GCS: {e}")
        raise

    return grouped_stats


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
