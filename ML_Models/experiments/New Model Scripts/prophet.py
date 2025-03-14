# MODEL PREDICTION
# PROPHET: 7-Day Demand Forecast per Product with RMSE & MAPE, and overall Average RMSE & MAPE

import numpy as np
from prophet import Prophet
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

    # Handle Date column
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

def train_prophet_model(product_data):
    prophet_df = product_data.rename(columns={'Date': 'ds', 'Total Quantity': 'y'})

    regressors = ['day_of_week', 'is_weekend', 'day_of_month', 'day_of_year', 'month', 'week_of_year']
    available_regressors = [col for col in regressors if col in prophet_df.columns]

    # Split the data
    train_df, test_df = train_test_split(prophet_df, test_size=0.2, shuffle=False)

    # Initialize the Prophet model
    model = Prophet(yearly_seasonality=True,
                   weekly_seasonality=True,
                   daily_seasonality=False)

    for regressor in available_regressors:
        model.add_regressor(regressor)

    model.fit(train_df)

    test_forecast = model.predict(test_df)

    y_true = test_df['y'].values
    y_pred = test_forecast['yhat'].values

    non_zero_indices = y_true != 0
    if sum(non_zero_indices) > 0:
        mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    else:
        mape = np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Create future dataframe for next 7 days forecast
    last_date = prophet_df['ds'].max()
    future = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)})

    # Add regressor values for future dates if they exist
    if 'day_of_week' in available_regressors:
        future['day_of_week'] = future['ds'].dt.dayofweek
    if 'is_weekend' in available_regressors:
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    if 'day_of_month' in available_regressors:
        future['day_of_month'] = future['ds'].dt.day
    if 'day_of_year' in available_regressors:
        future['day_of_year'] = future['ds'].dt.dayofyear
    if 'month' in available_regressors:
        future['month'] = future['ds'].dt.month
    if 'week_of_year' in available_regressors:
        future['week_of_year'] = future['ds'].dt.isocalendar().week.astype(int)

    # Make forecast
    forecast = model.predict(future)

    return {
        'rmse': rmse,
        'mape': mape,
        'forecast': forecast[['ds', 'yhat']]
    }

def suppress_logs():
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

    class NullWriter:
        def write(self, arg):
            pass
        def flush(self):
            pass

def main():
    try:
        suppress_logs()

        df = load_and_preprocess_data('ML.csv')

        products = df['Product Name'].unique()

        all_rmse = {}
        all_mape = {}
        all_forecasts = {}

        # Train model for each product
        for product in products:
            print(f"\nProcessing product: {product}")

            product_data = df[df['Product Name'] == product].copy()

            if len(product_data) < 10:
                print(f"Insufficient data for product {product}. Skipping.")
                continue

            # Train model and get results
            results = train_prophet_model(product_data)

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

