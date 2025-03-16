import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("C:/Users/svaru/Downloads/processed_good_transactions_20190103_20241231.csv")

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by Date and Product Name
df = df.sort_values(by=['Product Name', 'Date'])

# Encode Product Name
le = LabelEncoder()
df['Product_ID'] = le.fit_transform(df['Product Name'])

# Create lag features for past sales
lag_days = [7, 14, 21]
for lag in lag_days:
    df[f'lag_{lag}'] = df.groupby('Product Name')['Total Quantity'].shift(lag)

# Extract time-based features
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['week_of_year'] = df['Date'].dt.isocalendar().week
df['day_of_year'] = df['Date'].dt.dayofyear

# Drop rows with NaNs (from lag features)
df = df.dropna()

# Define features and target
features = ['Product_ID', 'day_of_week', 'month', 'week_of_year', 'day_of_year'] + [f'lag_{lag}' for lag in lag_days]
target = 'Total Quantity'

# Split data into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=False)

# Initialize LightGBM regressor
lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse', boosting_type='gbdt', verbose=-1)

# Define parameter grid for GridSearchCV
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'min_child_samples': [20, 50, 100]
}

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(train_df[features], train_df[target])

# Output best parameters
print(f'Best parameters found: {grid_search.best_params_}')

# Train model with best parameters on full training data
best_model = grid_search.best_estimator_

# Make predictions on validation set
valid_preds = best_model.predict(valid_df[features])
rmse = np.sqrt(mean_squared_error(valid_df[target], valid_preds))
print(f'Validation RMSE: {rmse}')

# Generate forecasts for the next 7 days
future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=7)
future_predictions = []

for product in df['Product Name'].unique():
    last_data = df[df['Product Name'] == product].iloc[-1]
    
    for date in future_dates:
        future_features = {
            'Product_ID': le.transform([product])[0],
            'day_of_week': date.dayofweek,
            'month': date.month,
            'week_of_year': date.isocalendar()[1],
            'day_of_year': date.timetuple().tm_yday
        }
        for lag in lag_days:
            future_features[f'lag_{lag}'] = last_data[f'lag_{lag}'] if f'lag_{lag}' in last_data else 0
        
        pred_quantity = best_model.predict(pd.DataFrame([future_features]))[0]
        future_predictions.append({'Date': date, 'Product Name': product, 'Predicted Quantity': pred_quantity})

# Convert to DataFrame
forecast_df = pd.DataFrame(future_predictions)
print(forecast_df)
