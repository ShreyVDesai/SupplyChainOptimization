import polars as pl
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def extracting_time_series_and_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, computes additional time-series features:
      - day_of_week, is_weekend, etc.
      - lag_1, lag_7, rolling_mean_7 of 'Total Quantity'
    """
    try:
        if df.empty:
            logger.warning("Input DataFrame is empty, returning an empty DataFrame.")
            return pd.DataFrame(
                columns=[
                    "Date", "Product Name", "Total Quantity",
                    "day_of_week", "is_weekend", "day_of_month", "day_of_year",
                    "month", "week_of_year", "lag_1", "lag_7", "rolling_mean_7"
                ]
            )

        # Ensure Date column is datetime type for feature extraction
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

            df["day_of_week"] = df["Date"].dt.weekday
            df["is_weekend"] = (df["Date"].dt.weekday > 5).astype(int)
            df["day_of_month"] = df["Date"].dt.day
            df["day_of_year"] = df["Date"].dt.dayofyear
            df["month"] = df["Date"].dt.month
            df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        else:
            logger.warning("Date column not found, skipping datetime feature extraction.")
            return df
    except Exception as e:
        logger.error(f"Error extracting datetime features during feature engineering: {e}")
        raise e

    try:
        # Only proceed with time series features if we have Total Quantity
        if "Total Quantity" in df.columns:
            # Sort by (Product Name, Date) for coherent time series ordering
            # df = df.sort_values(["Product Name", "Date"])

            df["lag_1"] = df.groupby("Product Name")["Total Quantity"].shift(1)
            df["lag_7"] = df.groupby("Product Name")["Total Quantity"].shift(7)
            df["rolling_mean_7"] = df.groupby("Product Name")["Total Quantity"].transform(lambda x: x.rolling(7, min_periods=1).mean())

        else:
            logger.warning("Total Quantity column not found, skipping lagged features")
            raise KeyError
    except Exception as e:
        logger.error(f"Error calculating lagged features during feature engineering: {e}")
        raise e

    return df