import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

# Hyperparameters
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
NUM_PRODUCTS = 10
BASE_DEMAND_RANGE = (100, 500)  # Adjusted for a broader range of daily sales volumes
NOISE_STD_RANGE = (10, 30)  # Reflecting typical daily sales fluctuations
STOCK_TICKERS = [
    "SPY",  # S&P 500 ETF
    "QQQ",  # Nasdaq-100 ETF
    "DIA",  # Dow Jones Industrial Average ETF
    "VTI",  # Total Stock Market ETF
    "VEU",  # FTSE All-World ex-US ETF
    "EFA",  # MSCI EAFE ETF
    "EEM",  # MSCI Emerging Markets ETF
    "IWV",  # Russell 3000 ETF
    "SCHX",  # Schwab U.S. Large-Cap ETF
    "IXUS",  # Total International Stock ETF
]
STOCK_INFLUENCE_STRENGTH = 0.85
WEEKLY_VARIATION_RANGE = (
    0.8,
    1.2,
)
SHOCK_MAGNITUDE_RANGE = (
    0.9,
    1.3,
)
SHOCK_PROBABILITY = 0.1  # Shocks for weekends


def generate_synthetic_data(
    start_date=START_DATE,
    end_date=END_DATE,
    num_products=NUM_PRODUCTS,
    base_demand_range=BASE_DEMAND_RANGE,
    noise_std_range=NOISE_STD_RANGE,
    stock_tickers=STOCK_TICKERS,
    stock_influence_strength=STOCK_INFLUENCE_STRENGTH,
    weekly_variation_range=WEEKLY_VARIATION_RANGE,
    shock_magnitude_range=SHOCK_MAGNITUDE_RANGE,
    shock_probability=SHOCK_PROBABILITY,
):
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    num_days = len(dates)

    # Assign a random stock ticker to each product
    product_stocks = np.random.choice(stock_tickers, size=num_products, replace=True)

    # Fetch stock market data
    stock_data = {}
    for ticker in set(product_stocks):
        stock_data[ticker] = (
            yf.download(ticker, start=start_date, end=end_date)["Close"]
            .reindex(dates)
            .ffill()
        )

    # Base demand per product
    base_demand = np.random.randint(*base_demand_range, size=num_products)

    # Noise: Random variations in demand
    noise_levels = np.random.uniform(*noise_std_range, size=num_products)
    noise = np.random.normal(0, noise_levels, size=(num_days, num_products))

    # Weekly demand fluctuations
    weekly_factors = np.tile(
        np.random.uniform(*weekly_variation_range, size=(7, num_products)),
        (num_days // 7 + 1, 1),
    )[:num_days]

    # Sudden demand shocks on weekends
    shock_effects = np.ones((num_days, num_products))
    weekend_indices = np.where(dates.dayofweek >= 5)[0]
    weekend_shocks = np.random.rand(len(weekend_indices)) < shock_probability
    shock_magnitudes = np.random.uniform(
        *shock_magnitude_range, size=len(weekend_indices)
    )
    for i, idx in enumerate(weekend_indices):
        if weekend_shocks[i]:
            shock_effects[idx, :] *= shock_magnitudes[i]

    # Compute final demand matrix
    demand_matrix = np.zeros((num_days, num_products))
    for i in range(num_products):
        stock_prices = stock_data[product_stocks[i]]
        stock_prices_normalized = stock_prices / stock_prices.mean()
        stock_influence = (
            stock_prices_normalized.values.reshape(-1) * stock_influence_strength
        )
        demand_matrix[:, i] = (
            base_demand[i]
            * (1 + stock_influence)
            * weekly_factors[:, i]
            * shock_effects[:, i]
            + noise[:, i]
        )

    # Ensure no negative values
    demand_matrix = np.maximum(demand_matrix, 0)

    # Create DataFrame with product-stock mapping in column names
    columns = [f"Product_{i+1} ({product_stocks[i]})" for i in range(num_products)]
    df = pd.DataFrame(demand_matrix, index=dates, columns=columns)

    return df, base_demand, noise_levels


def visualize_data(df, base_demand, noise_levels):
    """Plot the generated demand data."""
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(df.columns):
        plt.plot(df.index, df[column], label=f"{column}", alpha=0.7)
    plt.title("Synthetic Demand Driven by Stock Market")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    legend_labels = [
        f"{column} | Base: {base_demand[i]}, Noise STD: {noise_levels[i]:.2f}"
        for i, column in enumerate(df.columns)
    ]
    plt.legend(
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Product Info",
        fontsize=8,
    )

    # Add product metadata as text at the bottom of the graph
    metadata_text = "\n".join(
        [
            f"{column}: Base={base_demand[i]}, Noise STD={noise_levels[i]:.2f}"
            for i, column in enumerate(df.columns)
        ]
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_to_excel(df, start_date, end_date):
    """Save generated data to an Excel file."""
    save_dir = "./data/"
    os.makedirs(save_dir, exist_ok=True)
    filename = (
        f"stock_demand_{start_date.replace('-', '')}_{end_date.replace('-', '')}.xlsx"
    )
    filepath = os.path.join(save_dir, filename)

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=True)

    print(f"Data saved successfully to: {filepath}")


# Example usage
if __name__ == "__main__":
    synthetic_data, base_demand, noise_levels = generate_synthetic_data()
    visualize_data(synthetic_data, base_demand, noise_levels)
    save_to_excel(synthetic_data, START_DATE, END_DATE)
