import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import warnings

warnings.filterwarnings("ignore")


# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    return df


# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


# Train LSTM model for each product
def train_lstm_model(product_data):
    product_data = product_data[["Date", "Total Quantity"]].copy()
    product_data.sort_values(by="Date", inplace=True)

    # Normalize data
    scaler = MinMaxScaler()
    product_data["Total Quantity"] = scaler.fit_transform(
        product_data[["Total Quantity"]]
    )

    # Split dataset (70% train, 15% validation, 15% test)
    train_size = int(len(product_data) * 0.7)
    val_size = int(len(product_data) * 0.15)

    train_data = product_data.iloc[:train_size]["Total Quantity"].values
    val_data = product_data.iloc[train_size : train_size + val_size][
        "Total Quantity"
    ].values
    test_data = product_data.iloc[train_size + val_size :][
        "Total Quantity"
    ].values

    # Create sequences
    seq_length = 10  # Number of past days used for forecasting
    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(val_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Reshape for LSTM
    X_train = X_train.reshape(-1, seq_length, 1)
    X_val = X_val.reshape(-1, seq_length, 1)
    X_test = X_test.reshape(-1, seq_length, 1)

    # Build LSTM model
    model = Sequential(
        [
            LSTM(
                50,
                activation="relu",
                return_sequences=True,
                input_shape=(seq_length, 1),
            ),
            LSTM(50, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=0,
    )

    # Predict test set
    y_pred = model.predict(X_test).flatten()
    y_test = y_test.flatten()

    # Compute Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Forecast next 7 days
    future_input = test_data[-seq_length:].reshape(1, seq_length, 1)
    forecast = []
    for _ in range(7):
        pred = model.predict(future_input)[0, 0]
        forecast.append(pred)
        future_input = np.roll(future_input, -1)
        future_input[0, -1, 0] = pred

    # Inverse transform the forecast values
    forecast = scaler.inverse_transform(
        np.array(forecast).reshape(-1, 1)
    ).flatten()

    return {"rmse": rmse, "mape": mape, "r2": r2, "forecast": forecast}


def main():
    df = load_and_preprocess_data(
        "/Users/bhat/Desktop/SupplyChainOptimization/Data/cleaned_data.csv"
    )

    products = df["Product Name"].unique()
    print(f"Found {len(products)} unique products in the dataset.")

    all_metrics = {"RMSE": {}, "MAPE": {}, "R2": {}}
    all_forecasts = {}

    for product in products:
        print(f"\nProcessing product: {product}")

        product_data = df[df["Product Name"] == product].copy()
        if len(product_data) < 20:
            print(f"Insufficient data for {product}. Skipping.")
            continue

        results = train_lstm_model(product_data)

        all_metrics["RMSE"][product] = results["rmse"]
        all_metrics["MAPE"][product] = results["mape"]
        all_metrics["R2"][product] = results["r2"]
        all_forecasts[product] = results["forecast"]

        print(f"RMSE: {results['rmse']:.2f}")
        print(f"MAPE: {results['mape']:.2%}")
        print(f"R² Score: {results['r2']:.3f}")

        print("\nForecast for next 7 days:")
        for i, value in enumerate(results["forecast"]):
            print(f"Day {i+1}: {round(value, 2)}")

    # Average Metrics
    valid_rmse = [v for v in all_metrics["RMSE"].values() if not np.isnan(v)]
    valid_mape = [v for v in all_metrics["MAPE"].values() if not np.isnan(v)]
    valid_r2 = [v for v in all_metrics["R2"].values() if not np.isnan(v)]

    avg_rmse = sum(valid_rmse) / len(valid_rmse) if valid_rmse else np.nan
    avg_mape = sum(valid_mape) / len(valid_mape) if valid_mape else np.nan
    avg_r2 = sum(valid_r2) / len(valid_r2) if valid_r2 else np.nan

    print("\n--- Overall Model Performance ---")
    print(
        f"Average RMSE: {avg_rmse:.2f}"
        if not np.isnan(avg_rmse)
        else "Average RMSE: N/A"
    )
    print(
        f"Average MAPE: {avg_mape:.2%}"
        if not np.isnan(avg_mape)
        else "Average MAPE: N/A"
    )
    print(
        f"Average R² Score: {avg_r2:.3f}"
        if not np.isnan(avg_r2)
        else "Average R² Score: N/A"
    )


if __name__ == "__main__":
    main()
