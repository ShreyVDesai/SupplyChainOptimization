# MODEL PREDICTION
# SARIMA: 7-Day Demand Forecast per Product with RMSE & MAPE, and overall Average RMSE & MAPE
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import logging
import sys

# Load the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Date' and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
    except Exception as e:
        print(f"Error with mixed format: {e}")
        try:
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        except Exception as e:
            print(f"Error with infer_datetime_format: {e}")
            date_formats = ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']
            for fmt in date_formats:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                    print(f"Converted using format: {fmt}")
                    break
                except:
                    continue

    return df

def check_stationarity(time_series):
    """
    Perform Augmented Dickey-Fuller test to check stationarity

    Returns:
    - is_stationary: Boolean indicating if the series is stationary
    - p_value: p-value from the test
    - critical_values: Dictionary of critical values at different significance levels
    """
    # Perform ADF test
    result = adfuller(time_series.dropna())

    # Extract results
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Determine if stationary (p < 0.05)
    is_stationary = p_value < 0.05

    return {
        'is_stationary': is_stationary,
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values
    }

def train_sarima_model(product_data):
    # Set the date as index for time series analysis
    product_data = product_data.set_index('Date')

    y = product_data['Total Quantity']

    # Split the data
    train_size = int(len(y) * 0.8)
    train, test = y[:train_size], y[train_size:]

    # Try to find best parameters (simplified)
    order = (1, 1, 1)  # (p, d, q) - non-seasonal components
    seasonal_order = (1, 1, 1, 7)  # (P, D, Q, s) - seasonal components with weekly seasonality

    try:
        # Fit SARIMA model
        model = SARIMAX(train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        results = model.fit(disp=False, maxiter=100)

        predictions = results.get_forecast(steps=len(test))
        y_pred = predictions.predicted_mean

        y_true = test.values

        non_zero_indices = y_true != 0
        if sum(non_zero_indices) > 0:
            mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred.values[non_zero_indices]) / y_true[non_zero_indices])) * 100
        else:
            mape = np.nan

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Forecast next 7 days
        forecast = results.get_forecast(steps=7)
        forecast_mean = forecast.predicted_mean

        last_date = product_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_mean.values
        })

        return {
            'rmse': rmse,
            'mape': mape,
            'forecast': forecast_df
        }

    except Exception as e:
        print(f"Error training SARIMA model: {str(e)}")
        return {
            'rmse': np.nan,
            'mape': np.nan,
            'forecast': pd.DataFrame(columns=['ds', 'yhat'])
        }

def main():
    try:
        df = load_and_preprocess_data('ML.csv')

        products = df['Product Name'].unique()

        all_rmse = {}
        all_mape = {}
        all_forecasts = {}

        for product in products:
            print(f"\nProcessing product: {product}")

            # Filter data for this product
            product_data = df[df['Product Name'] == product].copy()

            if len(product_data) < 10:
                print(f"Insufficient data for product {product}. Skipping.")
                continue

            # Train model and get results
            results = train_sarima_model(product_data)

            all_rmse[product] = results['rmse']
            all_mape[product] = results['mape']
            all_forecasts[product] = results['forecast']

            print(f"RMSE: {results['rmse']:.2f}")
            if not np.isnan(results['mape']):
                print(f"MAPE: {results['mape']:.2f}%")
            else:
                print("MAPE: N/A (zero values in test set)")

            # Print forecast
            print("\nForecast for next 7 days:")
            for _, row in results['forecast'].iterrows():
                print(f"Date: {row['ds'].strftime('%Y-%m-%d')}, Forecast Quantity: {max(0, round(row['yhat'], 2))}")

        # Calculate average metrics across all products
        valid_rmse = [v for v in all_rmse.values() if not np.isnan(v)]
        valid_mape = [v for v in all_mape.values() if not np.isnan(v)]

        avg_rmse = sum(valid_rmse) / len(valid_rmse) if valid_rmse else np.nan
        avg_mape = sum(valid_mape) / len(valid_mape) if valid_mape else np.nan

        print("\n Overall Model Performance")
        if not np.isnan(avg_rmse):
            print(f"Average RMSE across all products: {avg_rmse:.2f}")
        else:
            print("Average RMSE: N/A")

        if not np.isnan(avg_mape):
            print(f"Average MAPE across all products: {avg_mape:.2f}%")
        else:
            print("Average MAPE: N/A")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

