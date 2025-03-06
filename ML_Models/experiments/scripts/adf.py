
import pandas as pd
import mlflow
from statsmodels.tsa.stattools import adfuller

mlflow.set_tracking_uri("file:///C:/Users/shrey/Projects/SupplyChainOptimization/ML_Models/experiments/mlruns")
mlflow.set_experiment("SARIMA_Preprocessing")

def load_data(csv_path):
    """Load time series data from a CSV file and aggregate by date to ensure unique index."""
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    product = 'beef'
    df = df[df['Product Name'] == product]
    # Group by Date and sum 'Total Quantity' to remove duplicates
    df = df.groupby(df.index)["Total Quantity"].sum().to_frame()

    # Ensure frequency is set (daily 'D' assumed here)
    df = df.asfreq('D', fill_value=0)  # Fill missing dates with 0

    return df

def adf_test(series):
    """Performs Augmented Dickey-Fuller test and returns results."""
    result = adfuller(series.dropna())  # Drop NaNs before testing
    return result[0], result[1]  # ADF Statistic, p-value

def preprocess_data(csv_path):
    """
    Loads data from CSV, filters the target variable, performs ADF test, logs results, and returns the processed DataFrame.
    """
    mlflow.set_experiment("SARIMA_Preprocessing")

    with mlflow.start_run(nested = True):
        # Load Data
        df = load_data(csv_path)

        # Perform ADF Test
        adf_stat, p_value = adf_test(df["Total Quantity"])

        # Log ADF results in MLflow
        mlflow.log_metric("ADF_Statistic", adf_stat)
        mlflow.log_metric("ADF_p_value", p_value)

        # Determine Stationarity
        stationarity = "Yes" if p_value < 0.05 else "No"
        mlflow.log_param("Stationarity", stationarity)

        print(f"ADF Test Completed - p-value: {p_value}")

    # **Explicitly End MLflow Run**
    mlflow.end_run()

    return df  # Return processed DataFrame for SARIMA

