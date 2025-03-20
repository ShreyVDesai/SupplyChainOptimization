import pandas as pd
import numpy as np
from math import sqrt
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap

def compute_rmse(y_true, y_pred):
    """Computes Root Mean Squared Error."""
    return sqrt(mean_squared_error(y_true, y_pred))

# =======================
# 1. Feature Engineering
# =======================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features for each row.
    Adds date features, lag features (lag_1, lag_7, lag_14, lag_30),
    and rolling statistics (rolling_mean_7, rolling_mean_14, rolling_std_7).
    You can add additional long-window features (e.g., 60-day rolling mean) as needed.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
    
    # Date-based features
    df["day_of_week"] = df["Date"].dt.dayofweek      # Monday=0, Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["day_of_month"] = df["Date"].dt.day
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    
    # Lag features
    df["lag_1"] = df.groupby("Product Name")["Total Quantity"].shift(1)
    df["lag_7"] = df.groupby("Product Name")["Total Quantity"].shift(7)
    df["lag_14"] = df.groupby("Product Name")["Total Quantity"].shift(14)
    df["lag_30"] = df.groupby("Product Name")["Total Quantity"].shift(30)
    
    # Rolling window features (short-term)
    df["rolling_mean_7"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    df["rolling_mean_14"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
    df["rolling_std_7"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).std())  # You can change window size if desired.
    
    # (Optional) Add longer-window features:
    # df["rolling_mean_60"] = df.groupby("Product Name")["Total Quantity"].transform(
    #     lambda x: x.shift(1).rolling(window=60, min_periods=1).mean())
    # df["rolling_std_60"] = df.groupby("Product Name")["Total Quantity"].transform(
    #     lambda x: x.shift(1).rolling(window=60, min_periods=1).std())
    
    # Additional features can be added here (difference, pct change, Fourier terms, etc.)
    
    return df

def create_target(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Create an aggregated target as the average of the next `horizon` days' 'Total Quantity'.
    This means each record's target is the average demand over the next 7 days.
    """
    df = df.sort_values(by=["Product Name", "Date"]).reset_index(drop=True)
    df["target"] = df.groupby("Product Name")["Total Quantity"].transform(
        lambda x: x.shift(-1).rolling(window=horizon, min_periods=1).mean()
    )
    return df

# ==========================================
# 2. Automatic Train/Validation/Test Split
# ==========================================

def get_train_valid_test_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.1):
    """
    Automatically splits the DataFrame into train, validation, and test sets based on the date range.
    """
    unique_dates = df["Date"].drop_duplicates().sort_values()
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
    param = {
        "objective": "reg:squarederror",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "random_state": 42
    }
    
    model = XGBRegressor(**param)
    model.fit(train_df[feature_columns], train_df[target_column])
    
    valid_pred = model.predict(valid_df[feature_columns])
    rmse = compute_rmse(valid_df[target_column], valid_pred)
    print(f"Trial {trial.number}: RMSE = {rmse}")
    return rmse

def hyperparameter_tuning(train_df, valid_df, feature_columns, target_column, n_trials: int = 100):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_df, valid_df, feature_columns, target_column), n_trials=n_trials)
    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    return trial.params

# ==========================================
# 4. Model Training Function (without early_stopping_rounds)
# ==========================================

def model_training(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_columns: list, target_column: str, params: dict = None):
    """
    Trains an XGBoost one-step forecasting model using the specified features and target.
    Uses an eval_set to display loss during training.
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
    eval_set = [(train_df[feature_columns], train_df[target_column]),
                (valid_df[feature_columns], valid_df[target_column])]
    model.fit(train_df[feature_columns],
              train_df[target_column],
              eval_set=eval_set,
              verbose=True)
    return model

# ==========================================
# 5. Iterative Forecasting Function (Updated to include dummy features)
# ==========================================

def iterative_forecast(model, df_product: pd.DataFrame, forecast_days: int = 7, product_columns: list = None) -> pd.DataFrame:
    """
    Given a trained one-step forecasting model and historical data for a single product,
    iteratively forecast the next `forecast_days` days.
    
    If `product_columns` (list of dummy feature names) is provided, they will be added
    to the feature set for prediction.
    Returns a DataFrame with forecast dates and predicted quantities.
    """
    history = df_product.copy().sort_values(by="Date").reset_index(drop=True)
    last_date = history["Date"].iloc[-1]
    forecasts = []
    
    base_feature_columns = [
        "lag_1", "lag_7", "lag_14", "lag_30", 
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "day_of_week", "is_weekend", "day_of_month", "day_of_year", "month", "week_of_year"
    ]
    
    for _ in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=1)
        feature_row = {
            "day_of_week": next_date.dayofweek,
            "is_weekend": int(next_date.dayofweek in [5, 6]),
            "day_of_month": next_date.day,
            "day_of_year": next_date.timetuple().tm_yday,
            "month": next_date.month,
            "week_of_year": next_date.isocalendar().week,
        }
        last_qty = history["Total Quantity"].iloc[-1]
        feature_row["lag_1"] = last_qty
        feature_row["lag_7"] = history.iloc[-7]["Total Quantity"] if len(history) >= 7 else last_qty
        feature_row["lag_14"] = history.iloc[-14]["Total Quantity"] if len(history) >= 14 else last_qty
        feature_row["lag_30"] = history.iloc[-30]["Total Quantity"] if len(history) >= 30 else last_qty
        
        qty_list = history["Total Quantity"].tolist()
        feature_row["rolling_mean_7"] = np.mean(qty_list[-7:]) if len(qty_list) >= 7 else np.mean(qty_list)
        feature_row["rolling_mean_14"] = np.mean(qty_list[-14:]) if len(qty_list) >= 14 else np.mean(qty_list)
        feature_row["rolling_std_7"] = np.std(qty_list[-7:]) if len(qty_list) >= 7 else np.std(qty_list)
        
        X_pred = pd.DataFrame([feature_row])[base_feature_columns]
        # Add product dummy features if provided:
        if product_columns is not None:
            current_product = df_product["Product Name"].iloc[0]
            dummy_features = {col: 0 for col in product_columns}
            product_dummy = f"prod_{current_product}"
            if product_dummy in dummy_features:
                dummy_features[product_dummy] = 1
            else:
                print(f"Warning: {product_dummy} not found in product_columns")
            for key, value in dummy_features.items():
                X_pred[key] = value
        
        next_qty = model.predict(X_pred)[0]
        # Round the predicted quantity to the nearest integer.
        next_qty = np.round(next_qty)
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


def save_model(model):
    with open("final_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved as final_model.pkl")




# ==========================================
# 6. Main Pipeline
# ==========================================


def main():
    # 1. Load data
    df = pd.read_csv("Data.csv")
    
    # 2. Create features and target on original data (keep for forecasting)
    original_df = extract_features(df.copy())
    original_df = create_target(original_df, horizon=7)
    original_df = original_df.dropna().reset_index(drop=True)
    
    # For training, work on a copy and then one-hot encode "Product Name"
    df_train = extract_features(df.copy())
    df_train = create_target(df_train, horizon=7)
    df_train = df_train.dropna().reset_index(drop=True)
    
    # -------------------------------
    # New step: Preserve original product name for evaluation
    df_train["Product"] = df_train["Product Name"]
    # -------------------------------
    
    df_train = pd.get_dummies(df_train, columns=["Product Name"], prefix="prod")
    
    # Update feature_columns to include one-hot encoded product columns.
    product_columns = [col for col in df_train.columns if col.startswith("prod_")]
    other_features = [
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "day_of_week", "is_weekend", "day_of_month", "day_of_year", "month", "week_of_year"
    ]
    feature_columns = other_features + product_columns
    target_column = "target"
    
    # ----- Step 2: Automatic Time-Based Train/Validation/Test Split -----
    train_df, valid_df, test_df = get_train_valid_test_split(df_train, train_frac=0.7, valid_frac=0.1)
    print(train_df)
    
    # ----- Step 3: Hyperparameter Tuning using Optuna -----
    print("Starting hyperparameter tuning with Optuna...")
    best_params = hyperparameter_tuning(train_df, valid_df, feature_columns, target_column, n_trials=5)
    
    # ----- Step 4: Train Final Model on Train+Validation -----
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    final_model = model_training(train_valid_df, valid_df, feature_columns, target_column, params=best_params)

    # Initialize SHAP explainer (TreeExplainer is used for tree-based models like XGBoost)
    explainer = shap.TreeExplainer(final_model)

    # Calculate SHAP values for the validation set
    shap_values = explainer.shap_values(valid_df[feature_columns])

    # Plot summary plot (global feature importance)
    shap.summary_plot(shap_values, valid_df[feature_columns])

    # Plot a SHAP value for a single prediction (local explanation)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], valid_df[feature_columns].iloc[0])
    
    # Compute and print training, validation, and test RMSE.
    train_rmse = compute_rmse(train_df[target_column], final_model.predict(train_df[feature_columns]))
    valid_rmse = compute_rmse(valid_df[target_column], final_model.predict(valid_df[feature_columns]))
    test_pred = final_model.predict(test_df[feature_columns])
    test_rmse = compute_rmse(test_df[target_column], test_pred)
    print("Training RMSE:", train_rmse)
    print("Validation RMSE:", valid_rmse)
    print("Test RMSE:", test_rmse)
    
    # -------------------------------
    # New step: Calculate RMSE for each product in the test set.
    test_df = test_df.copy()
    test_df["predicted"] = test_pred
    # 'Product' column was preserved before one-hot encoding.
    rmse_per_product = test_df.groupby("Product").apply(lambda d: compute_rmse(d[target_column], d["predicted"]))
    print("Individual RMSE per product:")
    print(rmse_per_product)
    # -------------------------------
    
    # ----- Step 5: Iterative Forecasting for Each Product -----
    # For iterative forecasting, use the original_df (which still contains the original "Product Name").
    products = original_df["Product Name"].unique()
    all_forecasts = []
    
    for product in products:
        df_product = original_df[original_df["Product Name"] == product].copy()
        if len(df_product) >= 60:
            fc = iterative_forecast(final_model, df_product, forecast_days=7, product_columns=product_columns)
            fc["Product Name"] = product
            all_forecasts.append(fc)
    
    all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    print("7-day forecasts for each product:")
    print(all_forecasts_df)
    
    all_forecasts_df.to_csv("7_day_forecasts.csv", index=False)
    save_model(final_model)

if __name__ == "__main__":
    main()
