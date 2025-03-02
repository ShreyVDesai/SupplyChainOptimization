import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Hyperparameters
START_DATE = "2019-01-03"
END_DATE = "2024-12-31"
BASE_DEMAND_RANGE = (
    100,
    500,
)  # Adjusted for a broader range of daily sales volumes
NOISE_STD_RANGE = (5, 15)  # Reflecting typical daily sales fluctuations
STOCK_INFLUENCE_STRENGTH = 0.5
WEEKLY_VARIATION_RANGE = (0.9, 1.1)
SHOCK_MAGNITUDE_RANGE = (0.95, 1.2)
SHOCK_PROBABILITY = 0.05  # Shocks for weekends

# Mapping of products to relevant commodity ETFs
PRODUCTS_TO_ETFS = {
    "Corn": "CORN",  # Teucrium Corn Fund
    "Wheat": "WEAT",  # Teucrium Wheat Fund
    "Soybeans": "SOYB",  # Teucrium Soybean Fund
    "Sugar": "CANE",  # Teucrium Sugar Fund
    "Coffee": "JO",  # iPath Series B Bloomberg Coffee Subindex Total Return ETN
    "Beef": "COW",  # iPath Series B Bloomberg Livestock Subindex Total Return ETN
    "Milk": "MOO",  # VanEck Agribusiness ETF (proxy for dairy)
    "Chocolate": "NIB",  # iPath Bloomberg Cocoa Subindex Total Return ETN
}


def generate_synthetic_data(
    start_date=START_DATE,
    end_date=END_DATE,
    base_demand_range=BASE_DEMAND_RANGE,
    noise_std_range=NOISE_STD_RANGE,
    products_to_etfs=PRODUCTS_TO_ETFS,
    stock_influence_strength=STOCK_INFLUENCE_STRENGTH,
    weekly_variation_range=WEEKLY_VARIATION_RANGE,
    shock_magnitude_range=SHOCK_MAGNITUDE_RANGE,
    shock_probability=SHOCK_PROBABILITY,
    fixed_base_demand=None,  # New parameter for user-specified demand
):
    """Generate synthetic demand data for products influenced by commodity ETFs, with optional fixed demand values."""

    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    num_days = len(dates)

    # Fetch stock market data for each ETF
    stock_data = {}
    for product, ticker in products_to_etfs.items():
        stock_prices = yf.download(ticker, start=start_date, end=end_date)[
            "Close"
        ]
        stock_data[product] = stock_prices.reindex(dates).ffill()

    # Base demand per product (fixed values take precedence)
    base_demand = {
        product: (
            fixed_base_demand.get(
                product, np.random.randint(*base_demand_range)
            )
            if fixed_base_demand
            else np.random.randint(*base_demand_range)
        )
        for product in products_to_etfs
    }

    # Noise: Random variations in demand
    noise_levels = {
        product: np.random.uniform(*noise_std_range)
        for product in products_to_etfs
    }
    noise = {
        product: np.random.normal(0, noise_levels[product], size=num_days)
        for product in products_to_etfs
    }

    # Weekly demand fluctuations
    weekly_factors = {
        product: np.tile(
            np.random.uniform(*weekly_variation_range, size=7),
            (num_days // 7 + 1),
        )[:num_days]
        for product in products_to_etfs
    }

    # Sudden demand shocks on weekends
    shock_effects = {
        product: np.ones(num_days) for product in products_to_etfs
    }
    weekend_indices = np.where(dates.dayofweek >= 5)[0]
    weekend_shocks = np.random.rand(len(weekend_indices)) < shock_probability
    shock_magnitudes = np.random.uniform(
        *shock_magnitude_range, size=len(weekend_indices)
    )
    for i, idx in enumerate(weekend_indices):
        if weekend_shocks[i]:
            for product in products_to_etfs:
                shock_effects[product][idx] *= shock_magnitudes[i]

    # Compute final demand matrix
    demand_data = {}
    for product in products_to_etfs:
        stock_prices = stock_data[product]
        stock_prices_normalized = stock_prices / stock_prices.mean()
        stock_influence = (
            stock_prices_normalized.values.reshape(-1)
            * stock_influence_strength
        )
        demand = (
            base_demand[product]
            * (1 + stock_influence)
            * weekly_factors[product]
            * shock_effects[product]
            + noise[product]
        )
        demand = np.maximum(demand, 0)  # Ensure no negative values
        demand_data[product] = np.round(demand).astype(int)

    # Create DataFrame
    df = pd.DataFrame(demand_data, index=dates)

    # Check for any missing values in the DataFrame
    if df.isnull().values.any():
        print(
            "Warning: Missing values detected in the DataFrame. Forward-filling missing data."
        )
        df.ffill(inplace=True)

    return df, base_demand, noise_levels


def visualize_data(df, base_demand, noise_levels):
    """Plot the generated demand data."""
    plt.figure(figsize=(12, 6))
    for product in df.columns:
        plt.plot(df.index, df[product], label=f"{product}", alpha=0.7)
    plt.title("Synthetic Demand Driven by Commodity ETFs")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    legend_labels = [
        f"{product} | Base: {base_demand[product]}, Noise STD: {noise_levels[product]:.2f}"
        for product in df.columns
    ]
    plt.legend(
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Product Info",
        fontsize=8,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_to_excel(df, start_date, end_date):
    """Save generated data to an Excel file."""
    save_dir = "../../data/"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"commodity_demand_{start_date.replace('-', '')}_{end_date.replace('-', '')}.xlsx"
    filepath = os.path.join(save_dir, filename)

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        df.to_excel(
            writer, sheet_name="Sheet1", index=True, index_label="Date"
        )

    print(f"Data saved successfully to: {filepath}")


# Example usage
if __name__ == "__main__":
    # Specify fixed demand values for some products (others remain random)
    fixed_base_demand = {
        "Corn": 75,
        "Wheat": 250,
        "Soybeans": 40,
        "Sugar": 60,
        "Coffee": 125,
        "Beef": 75,
        "Milk": 200,
        "Chocolate": 90,
    }

    synthetic_data, base_demand, noise_levels = generate_synthetic_data(
        fixed_base_demand=fixed_base_demand
    )

    visualize_data(synthetic_data, base_demand, noise_levels)
    save_to_excel(synthetic_data, START_DATE, END_DATE)
