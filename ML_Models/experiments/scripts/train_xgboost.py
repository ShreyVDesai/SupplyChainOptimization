import pandas as pd
import numpy as np
from math import sqrt
import optuna
from xgboost import XGBRegressor


# =======================
# 1. Feature Engineering
# =======================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features for each row.
    Adds date features, lag features (lag_1, lag_7, lag_14, lag_30),
    and rolling statistics (rolling_mean_7, rolling_mean_14, rolling_std_7).
    """
    # Ensure the Date column is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort by Product Name and Date for correct ordering
    df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
    
    # Create date-based features
    df["day_of_week"]   = df["Date"].dt.dayofweek  # Monday=0, Sunday=6
    df["is_weekend"]    = df["day_of_week"].isin([5,6]).astype(int)
    df["day_of_month"]  = df["Date"].dt.day
    df["day_of_year"]   = df["Date"].dt.dayofyear
    df["month"]         = df["Date"].dt.month
    df["week_of_year"]  = df["Date"].dt.isocalendar().week.astype(int)
    
    # Create lag features by grouping on Product Name
    df["lag_1"]   = df.groupby("Product Name")["Total Quantity"].shift(1)
    df["lag_7"]   = df.groupby("Product Name")["Total Quantity"].shift(7)
    df["lag_14"]  = df.groupby("Product Name")["Total Quantity"].shift(14)
    df["lag_30"]  = df.groupby("Product Name")["Total Quantity"].shift(30)
    
    # Create rolling features (shift by 1 to avoid leakage)
    df["rolling_mean_7"]  = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["rolling_mean_14"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )
    df["rolling_std_7"]   = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a one-step-ahead target: the next day’s 'Total Quantity'.
    """
    df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
    df["target"] = df.groupby("Product Name")["Total Quantity"].shift(-1)
    return df


# ==========================================
# 2. Automatic Train/Validation/Test Split
# ==========================================

def get_train_valid_test_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.1):
    """
    Automatically splits the DataFrame into train, validation, and test sets
    based on the date range in the data.
    
    The remaining fraction (1 - train_frac - valid_frac) is used for testing.
    """
    unique_dates = df["Date"]
    n = len(unique_dates)
    
    train_cutoff = unique_dates.iloc[int(n * train_frac)]
    valid_cutoff = unique_dates.iloc[int(n * (train_frac + valid_frac))]
    
    print("Train cutoff date:", train_cutoff)
    print("Validation cutoff date:", valid_cutoff)
    
    train_df = df[df["Date"] < train_cutoff].copy()
    valid_df = df[(df["Date"] >= train_cutoff) & (df["Date"] < valid_cutoff)].copy()
    test_df  = df[df["Date"] >= valid_cutoff].copy()
    
    return train_df, valid_df, test_df


# ==========================================
# 3. Hyperparameter Tuning using Optuna
# ==========================================

def objective(trial, train_df, valid_df, feature_columns, target_column):
    # Define hyperparameter search space
    param = {
        "objective": "reg:squarederror",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42
    }
    
    # Train on the training set
    model = XGBRegressor(**param)
    model.fit(train_df[feature_columns], train_df[target_column])
    
    # Predict on the validation set and calculate RMSE
    valid_pred = model.predict(valid_df[feature_columns])
    rmse = sqrt(mean_squared_error(valid_df[target_column], valid_pred))
    return rmse


def hyperparameter_tuning(train_df, valid_df, feature_columns, target_column, n_trials: int = 50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_df, valid_df, feature_columns, target_column), n_trials=n_trials)
    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    return trial.params


# ==========================================
# 4. Model Training Function
# ==========================================


def model_training(train_df: pd.DataFrame, feature_columns: list, target_column: str, params: dict = None):
    """
    Trains an XGBoost one-step forecasting model using the specified features and target.
    If params is provided, it uses those hyperparameters.
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    model = XGBRegressor(**params)
    model.fit(train_df[feature_columns], train_df[target_column])
    return model



# ==========================================
# 5. Iterative Forecasting Function
# ==========================================

def iterative_forecast(model, df_product: pd.DataFrame, forecast_days: int = 7) -> pd.DataFrame:
    """
    Given a trained one-step forecasting model and historical data for a single product,
    iteratively forecast the next `forecast_days` days.
    
    Returns a DataFrame with forecast dates and predicted quantities.
    """
    history = df_product.copy().sort_values(by="Date").reset_index(drop=True)
    last_date = history["Date"].iloc[-1]
    forecasts = []
    
    # Define feature columns (must match training features)
    feature_columns = [
        "lag_1", "lag_7", "lag_14", "lag_30", 
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "day_of_week", "is_weekend", "day_of_month", "day_of_year", "month", "week_of_year"
    ]
    
    for _ in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=1)
        feature_row = {}
        feature_row["day_of_week"]  = next_date.dayofweek
        feature_row["is_weekend"]   = int(next_date.dayofweek in [5, 6])
        feature_row["day_of_month"] = next_date.day
        feature_row["day_of_year"]  = next_date.timetuple().tm_yday
        feature_row["month"]        = next_date.month
        feature_row["week_of_year"] = next_date.isocalendar().week
        
        # Use historical values for lag features
        last_qty = history["Total Quantity"].iloc[-1]
        feature_row["lag_1"]  = last_qty
        feature_row["lag_7"]  = history.iloc[-7]["Total Quantity"] if len(history) >= 7 else last_qty
        feature_row["lag_14"] = history.iloc[-14]["Total Quantity"] if len(history) >= 14 else last_qty
        feature_row["lag_30"] = history.iloc[-30]["Total Quantity"] if len(history) >= 30 else last_qty
        
        qty_list = history["Total Quantity"].tolist()
        feature_row["rolling_mean_7"]  = np.mean(qty_list[-7:])  if len(qty_list) >= 7 else np.mean(qty_list)
        feature_row["rolling_mean_14"] = np.mean(qty_list[-14:]) if len(qty_list) >= 14 else np.mean(qty_list)
        feature_row["rolling_std_7"]   = np.std(qty_list[-7:])   if len(qty_list) >= 7 else np.std(qty_list)
        
        X_pred = pd.DataFrame([feature_row])[feature_columns]
        next_qty = model.predict(X_pred)[0]
        forecasts.append({"Date": next_date, "Predicted Quantity": next_qty})
        
        new_row = {
            "Date": next_date,
            "Product Name": df_product["Product Name"].iloc[0],
            "Total Quantity": next_qty,
            "lag_1": next_qty,
            "day_of_week": feature_row["day_of_week"],
            "is_weekend": feature_row["is_weekend"],
            "day_of_month": feature_row["day_of_month"],
            "day_of_year": feature_row["day_of_year"],
            "month": feature_row["month"],
            "week_of_year": feature_row["week_of_year"],
            "lag_7": feature_row["lag_7"],
            "lag_14": feature_row["lag_14"],
            "lag_30": feature_row["lag_30"],
            "rolling_mean_7": feature_row["rolling_mean_7"],
            "rolling_mean_14": feature_row["rolling_mean_14"],
            "rolling_std_7": feature_row["rolling_std_7"],
            "target": np.nan
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date
        
    return pd.DataFrame(forecasts)


def main():
    
    # 1. Load data
    df = pd.read_csv("Data.csv")

    # 2. Create features and target
    df = extract_features(df)
    df = create_target(df)

    # Drop rows with missing values (common at the beginning and end of time series)
    df = df.dropna().reset_index(drop=True)


    # ----- Step 2: Automatic Time-Based Train/Validation/Test Split -----
    train_df, valid_df, test_df = get_train_valid_test_split(df, train_frac=0.7, valid_frac=0.1)
    
    feature_columns = [
        "lag_1", "lag_7", "lag_14", "lag_30", 
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "day_of_week", "is_weekend", "day_of_month", "day_of_year", "month", "week_of_year"
    ]
    target_column = "target"

    pd.to_csv(train_df)


    # # ----- Step 3: Hyperparameter Tuning using Optuna -----
    # print("Starting hyperparameter tuning with Optuna...")
    # best_params = hyperparameter_tuning(train_df, valid_df, feature_columns, target_column, n_trials=50)
    
    # # ----- Step 4: Train Final Model on Train+Validation -----
    # train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    # final_model = model_training(train_valid_df, feature_columns, target_column, params=best_params)
    
    # # Evaluate on the test set
    # test_pred = final_model.predict(test_df[feature_columns])
    # test_rmse = sqrt(mean_squared_error(test_df[target_column], test_pred))
    # print("Test RMSE:", test_rmse)


    # # ----- Step 5: Iterative Forecasting for Each Product -----
    # products = df["Product Name"].unique()
    # all_forecasts = []



    # # Only forecast for products with at least 60 days of history
    # for product in products:
    #     df_product = df[df["Product Name"] == product].copy()
    #     if len(df_product) >= 60:
    #         fc = iterative_forecast(final_model, df_product, forecast_days=7)
    #         fc["Product Name"] = product
    #         all_forecasts.append(fc)
    
    # all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    # print("7-day forecasts for each product:")
    # print(all_forecasts_df)
    
    # # Optionally, save the forecasts to a CSV file
    # all_forecasts_df.to_csv("7_day_forecasts.csv", index=False)



