import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv("C:/Users/svaru/Downloads/processed_good_transactions_20190103_20241231.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Pivot Table
df_pivot = df.pivot(index='Date', columns='Product Name', values='Total Quantity').fillna(0)

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_pivot)

# Create Sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5  # Increased sequence length
X, y = create_sequences(scaled_data, seq_length)

# Define Improved LSTM Model
model = Sequential([
    Input(shape=(seq_length, X.shape[2])),
    Bidirectional(LSTM(100, activation='relu', return_sequences=True)),  # Bidirectional LSTM
    Dropout(0.2),  # Dropout for Regularization
    LSTM(100, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(X.shape[2])  # Output Layer
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train Model
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping, reduce_lr])

future_predictions = []
input_seq = X[-1]  # Start with the last sequence

for _ in range(7):
    pred = model.predict(input_seq.reshape(1, seq_length, X.shape[2]))[0]
    future_predictions.append(pred)
    
    # Roll window: Remove first element & add new prediction
    input_seq = np.roll(input_seq, shift=-1, axis=0)
    input_seq[-1] = pred

# Convert Predictions Back to Original Scale
future_predictions = scaler.inverse_transform(np.array(future_predictions))

# Compare with Actual Next 7 Days (if available)
actual_future_values = df_pivot.iloc[-7:].values  # Assuming we have actual data for the next 7 days

# RMSE Calculation
rmse = np.sqrt(mean_squared_error(actual_future_values, future_predictions))
print(f"7-Day Forecast RMSE: {rmse}")
