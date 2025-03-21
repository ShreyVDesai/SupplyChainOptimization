import numpy as np
import pandas as pd
import tensorflow as tf
import shap
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

def extracting_time_series_and_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create time-based features
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 if Saturday/Sunday, else 0

    # Sort by date for proper lag calculations
    df.sort_values(by=["Date", "Product Name"], inplace=True)

    # Create lag features
    df['lag_1'] = df.groupby('Product Name')['Total Quantity'].shift(1)  # Yesterday's value
    df['lag_7'] = df.groupby('Product Name')['Total Quantity'].shift(7)  # Last week's value
    
    # Rolling mean features (e.g., 7-day moving average)
    df['rolling_mean_7'] = df.groupby('Product Name')['Total Quantity'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Drop missing lag values (first few rows will be NaN)
    df.dropna(inplace=True)
    
    return df

# Load & Process Data
df = pd.read_csv("C:/Users/svaru/Downloads/processed_good_transactions_20190103_20241231.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = extracting_time_series_and_lagged_features(df)

# Pivot Table to get all product quantities as separate columns
df_pivot = df.pivot(index='Date', columns='Product Name', values=['Total Quantity', 'lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'is_weekend']).fillna(0)

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_pivot)

# Create Sequences (LSTMs require sequence data)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(scaled_data, seq_length)

# Train-Val Split
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]


# After model training, load the trained model
model = tf.keras.models.load_model("E:/MLOps/SupplyChainOptimization/ML_Models/experiments/scripts/LSTM_model.h5")

# Prepare SHAP Explainer
background_data = X_train[:100]  # Using the first 100 training samples as background for SHAP
explainer = shap.Explainer(model, background_data)

# Get SHAP values for a subset of the validation set (first 10 samples)
shap_values = explainer(X_val[:10])  # or use any part of the dataset you want to explain

# Visualize SHAP values - Global Feature Importance
shap.summary_plot(shap_values, X_val[:10], feature_names=df_pivot.columns)

# Visualize SHAP values - Local Explanation for the first validation sample
shap.force_plot(shap_values[0], X_val[0], feature_names=df_pivot.columns)
