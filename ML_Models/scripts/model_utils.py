import polars as pl


def extracting_time_series_and_lagged_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    For each row, computes additional time-series features:
      - day_of_week, is_weekend, etc.
      - lag_1, lag_7, rolling_mean_7 of 'Total Quantity'
    """
    try:
        if df.is_empty():
            return pl.DataFrame(
                schema={
                    "Date": pl.Date,
                    "Product Name": pl.Utf8,
                    "Total Quantity": pl.Float64,
                    "day_of_week": pl.Int32,
                    "is_weekend": pl.Int8,
                    "day_of_month": pl.Int32,
                    "day_of_year": pl.Int32,
                    "month": pl.Int32,
                    "week_of_year": pl.Int32,
                    "lag_1": pl.Float64,
                    "lag_7": pl.Float64,
                    "rolling_mean_7": pl.Float64,
                }
            )

        # Ensure Date column is datetime type for feature extraction
        if "Date" in df.columns:
            df = df.with_columns(pl.col("Date").cast(pl.Datetime))

            df = df.with_columns(
                pl.col("Date").dt.weekday().alias("day_of_week"),
                (pl.col("Date").dt.weekday() > 5)
                .cast(pl.Int8)
                .alias("is_weekend"),
                pl.col("Date").dt.day().alias("day_of_month"),
                pl.col("Date").dt.ordinal_day().alias("day_of_year"),
                pl.col("Date").dt.month().alias("month"),
                pl.col("Date").dt.week().alias("week_of_year"),
            )
        else:
            return df
    except Exception as e:
        raise e

    try:
        # Only proceed with time series features if we have Total Quantity
        if "Total Quantity" in df.columns:
            # Sort by (Product Name, Date) for coherent time series ordering
            df = df.sort(["Product Name", "Date"]).with_columns(
                [
                    pl.col("Total Quantity")
                    .shift(1)
                    .over("Product Name")
                    .alias("lag_1"),
                    pl.col("Total Quantity")
                    .shift(7)
                    .over("Product Name")
                    .alias("lag_7"),
                    pl.col("Total Quantity")
                    .rolling_mean(window_size=7)
                    .over("Product Name")
                    .alias("rolling_mean_7"),
                ]
            )
        else:
            raise KeyError
    except Exception as e:
        raise e

    return df
