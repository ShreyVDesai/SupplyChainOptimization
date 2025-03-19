import os
import random
import time
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Hyperparameters
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
BASE_DEMAND_RANGE = (100, 500)
NOISE_STD_RANGE = (5, 15)
STOCK_INFLUENCE_STRENGTH = 0.5
WEEKLY_VARIATION_RANGE = (0.9, 1.1)
SHOCK_MAGNITUDE_RANGE = (0.95, 1.2)
SHOCK_PROBABILITY = 0.05
REQUEST_DELAY = 1.8  # Delay in seconds between API requests


def fetch_nasdaq_symbols():
    """Fetches the list of NASDAQ company symbols from NASDAQ Trader."""
    url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    df = pd.read_csv(url, sep="|")
    # Remove footer row and test issues
    df = df[df["Symbol"] != "File Creation Time:"]
    df = df[df["Test Issue"] == "N"]
    return df["Symbol"].tolist()


def save_to_excel(df, start_date, end_date):
    """Save generated data to an Excel file."""
    save_dir = "./data/"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"nasdaq_stock_demand_{start_date.replace('-', '')}_{end_date.replace('-', '')}.xlsx"
    filepath = os.path.join(save_dir, filename)
    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=True, index_label="Date")
    print(f"Data saved successfully to: {filepath}")


def generate_synthetic_data(symbols, start_date=START_DATE, end_date=END_DATE):
    """
    Generate synthetic demand data influenced by stock prices.
    If an unexpected error occurs during the process, the existing (partial) data is saved.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    num_days = len(dates)
    stock_data = {}
    
    try:
        # Fetch stock market data for each symbol
        for symbol in symbols:
            try:
                stock_prices = yf.download(symbol, start=start_date, end=end_date)["Close"]
                # Reindex to ensure we have data for all dates and forward-fill missing values
                stock_data[symbol] = stock_prices.reindex(dates).ffill()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                # Continue with the next symbol without breaking the loop.
            time.sleep(REQUEST_DELAY)
        
        if not stock_data:
            raise Exception("No stock data could be fetched.")
        
        # Work only with the symbols that successfully fetched data.
        available_symbols = list(stock_data.keys())
        
        # Base demand per symbol
        base_demand = {symbol: np.random.randint(*BASE_DEMAND_RANGE) for symbol in available_symbols}
        # Noise: Random variations in demand
        noise_levels = {symbol: np.random.uniform(*NOISE_STD_RANGE) for symbol in available_symbols}
        noise = {symbol: np.random.normal(0, noise_levels[symbol], size=num_days) for symbol in available_symbols}
        
        # Weekly demand fluctuations
        weekly_factors = {
            symbol: np.tile(np.random.uniform(*WEEKLY_VARIATION_RANGE, size=7),
                             (num_days // 7 + 1))[:num_days]
            for symbol in available_symbols
        }
        
        # Sudden demand shocks on weekends
        shock_effects = {symbol: np.ones(num_days) for symbol in available_symbols}
        weekend_indices = np.where(dates.dayofweek >= 5)[0]
        weekend_shocks = np.random.rand(len(weekend_indices)) < SHOCK_PROBABILITY
        shock_magnitudes = np.random.uniform(*SHOCK_MAGNITUDE_RANGE, size=len(weekend_indices))
        for i, idx in enumerate(weekend_indices):
            if weekend_shocks[i]:
                for symbol in available_symbols:
                    shock_effects[symbol][idx] *= shock_magnitudes[i]
        
        # Compute final demand matrix
        demand_data = {}
        for symbol in available_symbols:
            stock_prices = stock_data[symbol]
            stock_prices_normalized = stock_prices / stock_prices.mean()
            stock_influence = stock_prices_normalized.values.reshape(-1) * STOCK_INFLUENCE_STRENGTH
            demand = (base_demand[symbol] * (1 + stock_influence) *
                      weekly_factors[symbol] * shock_effects[symbol] +
                      noise[symbol])
            demand = np.maximum(demand, 0)
            demand_data[symbol] = np.round(demand).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame(demand_data, index=dates)
        if df.isnull().values.any():
            print("Warning: Missing values detected in the DataFrame. Forward-filling missing data.")
            df.ffill(inplace=True)
        
        return df, base_demand, noise_levels
    
    except Exception as e:
        print("Unexpected error occurred. Saving partial data.")
        partial_df = pd.DataFrame(stock_data, index=dates)
        save_to_excel(partial_df, start_date, end_date)
        raise e


def visualize_data(df, base_demand, noise_levels):
    """Plot the generated demand data."""
    plt.figure(figsize=(12, 6))
    for symbol in df.columns:
        plt.plot(df.index, df[symbol], label=f"{symbol}", alpha=0.7)
    plt.title("Synthetic Demand Driven by NASDAQ Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    legend_labels = [
        f"{symbol} | Base: {base_demand[symbol]}, Noise STD: {noise_levels[symbol]:.2f}"
        for symbol in df.columns
    ]
    plt.legend(
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Symbol Info",
        fontsize=8,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    all_symbols = fetch_nasdaq_symbols()
    print(f"Fetched {len(all_symbols)} NASDAQ symbols.")
    if len(all_symbols) < 1000:
        print("Not enough symbols available to select 1000, using all available symbols.")
        selected_symbols = all_symbols
    else:
        selected_symbols = all_symbols, 1000

    synthetic_data, base_demand, noise_levels = generate_synthetic_data(selected_symbols)
    visualize_data(synthetic_data, base_demand, noise_levels)
    save_to_excel(synthetic_data, START_DATE, END_DATE)
