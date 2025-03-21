import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Function: Extract Lag & Time-Based Features
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
df = pd.read_csv("C:/Users/svaru/Downloads/test.csv")
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

# Visualize training and validation loss
def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss during Training')


# Hyperparameter Optimization with MLflow & Optuna
def objective(trial):
    with mlflow.start_run():
        lstm_units = trial.suggest_int("lstm_units", 50, 200, step=50)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 3)

        model = Sequential()
        model.add(Input(shape=(seq_length, X.shape[2])))

        for _ in range(num_layers):
            model.add(Bidirectional(LSTM(lstm_units, activation='relu', return_sequences=True)))
            model.add(Dropout(dropout_rate))

        model.add(LSTM(lstm_units // 2, activation='relu'))
        model.add(Dense(lstm_units // 2, activation='relu'))
        model.add(Dense(X.shape[2]))  # Multi-output: One for each product

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        history = model.fit(
            X_train, y_train,
            epochs=50, batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        plot_loss(history)

        model.save("LSTM_model.h5")
        with open('LSTM_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Log the model with MLflow
        mlflow.tensorflow.log_model(model, "model")


        # Predict Next 7 Days
        future_predictions = []
        input_seq = X[-1]  # Last known sequence

        for _ in range(7):
            pred = model.predict(input_seq.reshape(1, seq_length, X.shape[2]))[0]
            future_predictions.append(pred)
            input_seq = np.roll(input_seq, shift=-1, axis=0)
            input_seq[-1] = pred

        future_predictions = scaler.inverse_transform(np.array(future_predictions))
        actual_future_values = df_pivot.iloc[-7:].values

        rmse = np.sqrt(mean_squared_error(actual_future_values, future_predictions))

        # Save plot as an image
        plot_path = "train_val_loss.png"
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss during Training')
        plt.savefig(plot_path)

        mlflow.log_artifact(plot_path)


        mlflow.log_param("lstm_units", lstm_units)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_metric("rmse", rmse)

        return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

best_params = study.best_params
print("Best Parameters:", best_params)
print(f"Best RMSE: {study.best_value}")
