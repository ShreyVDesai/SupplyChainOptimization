import pandas as pd
import mlflow
import sys
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from adf import preprocess_data  # Import preprocessing function

mlflow.set_tracking_uri("file:///C:/Users/shrey/Projects/SupplyChainOptimization/ML_Models/experiments/mlruns")
mlflow.set_experiment("SARIMA_Demand_Forecasting")


def calculate_metrics(actual, predicted):
    """Calculate error metrics between actual and predicted values."""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # Mean Absolute Percentage Error (MAPE)

    return mae, mse, rmse, mape

# def train_sarima(csv_path):
#     """
#     Calls adf.py to get processed data, trains SARIMA, evaluates performance, and logs experiment in MLflow.
#     """
#     mlflow.set_experiment("SARIMA_Demand_Forecasting")

#     with mlflow.start_run(nested = True):
#         # Get preprocessed data
#         df = preprocess_data(csv_path)

#         # Select the 'Total Quantity' column for training
#         y = df["Total Quantity"]

#         # Auto ARIMA to find best parameters
#         auto_model = auto_arima(y, seasonal=True,  m=[7, 14, 30], trace=True, stepwise=True, d=2, D=1)

#         # Extract best parameters
#         order = auto_model.order
#         seasonal_order = auto_model.seasonal_order
#         print(f"Best Order: {order}, Seasonal Order: {seasonal_order}")

#         # Train SARIMA Model
#         model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
#                         enforce_stationarity=False, enforce_invertibility=False)
#         model_fit = model.fit(maxiter=200, disp=False)

#         # Forecasting
#         forecast_steps = 7  # Predict for the next 7 days
#         forecast = model_fit.forecast(steps=forecast_steps)

#         # Compare with actual values (if available)
#         if len(y) >= forecast_steps:
#             actual = y[-forecast_steps:]
#             mae, mse, rmse, mape = calculate_metrics(actual, forecast)

#             print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

#             # Log metrics in MLflow
#             mlflow.log_metric("MAE", mae)
#             mlflow.log_metric("MSE", mse)
#             mlflow.log_metric("RMSE", rmse)
#             mlflow.log_metric("MAPE", mape)

#         # Log SARIMA parameters
#         mlflow.log_param("order", order)
#         mlflow.log_param("seasonal_order", seasonal_order)
#         mlflow.log_metric("AIC", model_fit.aic)
#         mlflow.log_metric("BIC", model_fit.bic)

#         # Save trained model
#         mlflow.sklearn.log_model(model_fit, "SARIMA_Model")

#         print(f"Model trained with AIC: {model_fit.aic}, BIC: {model_fit.bic}")
def train_sarima(csv_path):
    """
    Calls adf.py to get processed data, trains SARIMA, evaluates performance, and logs experiment in MLflow.
    """
    mlflow.set_experiment("SARIMA_Demand_Forecasting")

    with mlflow.start_run(nested=True):
        # Get preprocessed data
        df = preprocess_data(csv_path)

        # Select the 'Total Quantity' column for training
        y = df["Total Quantity"]

        # Define possible seasonal periods
        seasonal_periods = [7, 14, 30]

        # Track the best model
        best_model = None
        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None
        best_m = None  # Store best m value

        for m in seasonal_periods:
            try:
                print(f"Trying seasonal period m={m}...")
                model_candidate = auto_arima(y, seasonal=True, m=m, trace=True, stepwise=True, d=2, D=1)

                if model_candidate.aic() < best_aic:
                    best_aic = model_candidate.aic()
                    best_model = model_candidate
                    best_order = model_candidate.order
                    best_seasonal_order = model_candidate.seasonal_order
                    best_m = m  # Store the best seasonal period

            except Exception as e:
                print(f"Auto ARIMA failed for m={m} with error: {e}")
                continue

        # If no model was selected, raise an error
        if best_model is None:
            raise ValueError("Auto ARIMA failed for all seasonal periods!")

        print(f"Best Order: {best_order}, Seasonal Order: {best_seasonal_order}, Best m: {best_m}")

        # Train SARIMA Model
        model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(maxiter=200, disp=False)

        # Forecasting
        forecast_steps = 7  # Predict for the next 7 days
        forecast = model_fit.forecast(steps=forecast_steps)

        # Compare with actual values (if available)
        if len(y) >= forecast_steps:
            actual = y[-forecast_steps:]
            mae, mse, rmse, mape = calculate_metrics(actual, forecast)

            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

            # Log metrics in MLflow
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAPE", mape)

        # Log SARIMA parameters
        mlflow.log_param("order", best_order)
        mlflow.log_param("seasonal_order", best_seasonal_order)
        mlflow.log_param("seasonal_period", best_m)
        mlflow.log_metric("AIC", model_fit.aic)
        mlflow.log_metric("BIC", model_fit.bic)

        # Save trained model
        mlflow.sklearn.log_model(model_fit, "SARIMA_Model")

        print(f"Model trained with AIC: {model_fit.aic}, BIC: {model_fit.bic}")


if __name__ == "__main__":
    csv_path = "C:/Users/shrey/Projects/SupplyChainOptimization/data/sales_data.csv"
    train_sarima(csv_path)
