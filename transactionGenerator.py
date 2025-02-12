import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import uuid
from tqdm import tqdm

# ============================ CONSTANTS ============================

# File Paths
SAVE_DIR = "./transaction/"
DEMAND_FILEPATH = "./data/commodity_demand_20190103_20241231.xlsx"

# Pricing & Transactions
BASE_PRICE_RANGE = (5.00, 50.00)  # Base price in USD
NUM_TRANSACTIONS_RANGE = (60, 80)  # Min & max transactions per product per day

# Store Locations (Predefined)
STORE_LOCATIONS = ["Downtown", "Uptown", "Westside", "Eastside", "Suburban"]

# Business Hours
START_TIME = "08:00:00"  # Store opens at 8 AM
END_TIME = "20:00:00"  # Store closes at 8 PM

# Producer & Product Settings
PRODUCER_ID_RANGE = (1, 11)  # Range for producer IDs

# Inflation Rates (Annual, in percentage)
INFLATION_RATES = {
    2000: 3.4,
    2001: 2.8,
    2002: 1.6,
    2003: 2.3,
    2004: 2.7,
    2005: 3.4,
    2006: 3.2,
    2007: 2.9,
    2008: 3.8,
    2009: -0.4,
    2010: 1.6,
    2011: 3.2,
    2012: 2.1,
    2013: 1.5,
    2014: 1.6,
    2015: 0.1,
    2016: 1.3,
    2017: 2.1,
    2018: 2.4,
    2019: 1.8,
    2020: 1.2,
    2021: 4.7,
    2022: 8.0,
    2023: 4.1,
    2024: 2.9,
}

# ============================ FUNCTIONS ============================


def load_demand_data(filepath):
    """Load the generated demand data from an Excel file."""
    df = pd.read_excel(filepath, index_col="Date", parse_dates=True)
    return df


def generate_base_prices(product_names, fixed_prices=None):
    """
    Generate a dictionary of base prices for each product.

    Args:
        product_names (list): List of product names from demand data.
        fixed_prices (dict): A dictionary where keys are product names and values are fixed starting prices.

    Returns:
        dict: A dictionary mapping product names to their base prices.
    """
    if fixed_prices is None:
        fixed_prices = {}

    base_prices = {}

    for product in product_names:
        if product in fixed_prices:
            base_prices[product] = round(fixed_prices[product], 2)
        else:
            base_prices[product] = round(np.random.uniform(*BASE_PRICE_RANGE), 2)

    return base_prices


def adjust_price_for_inflation(base_price, start_year, current_year):
    """Adjust the base price for inflation from the second year onward."""
    adjusted_price = base_price

    # Only apply inflation if we have moved past the start year
    if current_year > start_year:
        for year in range(start_year + 1, current_year + 1):
            if year in INFLATION_RATES:
                adjusted_price *= 1 + INFLATION_RATES[year] / 100
            else:
                raise ValueError(f"Inflation rate for year {year} is not available.")

    return round(adjusted_price, 2)


def generate_transactions(demand_df, fixed_prices=None):
    """Generate transaction data based on demand and adjust prices for inflation."""
    transactions = []
    product_names = list(demand_df.columns)  # Extract product names from demand data
    base_prices = generate_base_prices(product_names, fixed_prices)

    start_year = demand_df.index.min().year  # Determine earliest year in demand data

    # Initialize tqdm progress bar for the date iteration
    for date, row in tqdm(
        demand_df.iterrows(), total=demand_df.shape[0], desc="Processing transactions"
    ):
        current_year = date.year
        daily_transactions = []

        for product, demand in row.items():
            if demand <= 0:
                continue

            if product not in base_prices:
                raise KeyError(f"Product {product} not found in base prices.")

            # Adjust the base price for inflation
            base_price = base_prices[product]
            cost_price = adjust_price_for_inflation(
                base_price, start_year, current_year
            )

            # Determine number of transactions
            num_transactions = np.random.randint(*NUM_TRANSACTIONS_RANGE)
            quantities = np.random.multinomial(
                int(demand), np.ones(num_transactions) / num_transactions
            )

            for qty in quantities:
                if qty == 0:
                    continue

                daily_transactions.append(
                    [
                        date,  # Placeholder; will be updated with correct time
                        round(cost_price, 2),
                        str(uuid.uuid4()),  # Unique transaction ID
                        int(qty),
                        np.random.randint(*PRODUCER_ID_RANGE),  # Producer ID
                        np.random.choice(STORE_LOCATIONS),  # Store location
                        product,  # Product Name
                    ]
                )

        np.random.shuffle(daily_transactions)  # Shuffle transactions for the day

        # Assign more realistic timestamps (clustered around midday)
        num_transactions = len(daily_transactions)
        if num_transactions > 0:
            start_time = datetime.combine(
                date, datetime.strptime(START_TIME, "%H:%M:%S").time()
            )
            end_time = datetime.combine(
                date, datetime.strptime(END_TIME, "%H:%M:%S").time()
            )

            # Generate normally distributed time deltas centered around midday
            mean_time = start_time + timedelta(hours=4)  # Centered around 1 PM
            std_dev = 1.5  # 1.5-hour standard deviation for more natural variation

            random_offsets = np.random.normal(
                loc=4 * 3600, scale=std_dev * 3600, size=num_transactions
            )

            # Clip times to ensure they stay within the 8 AM - 8 PM window
            random_offsets = np.clip(
                random_offsets, 0, (end_time - start_time).total_seconds()
            )

            # Sort to maintain chronological order
            random_offsets = np.sort(random_offsets)

            for i, transaction in enumerate(daily_transactions):
                transaction[0] = start_time + timedelta(seconds=random_offsets[i])

        transactions.extend(daily_transactions)

    transactions_df = pd.DataFrame(
        transactions,
        columns=[
            "Date",
            "Cost Price",
            "Transaction ID",
            "Quantity",
            "Producer ID",
            "Store Location",
            "Product Name",
        ],
    )

    # Convert column data types explicitly
    transactions_df["Date"] = pd.to_datetime(transactions_df["Date"])
    transactions_df["Cost Price"] = transactions_df["Cost Price"].astype(float)
    transactions_df["Quantity"] = transactions_df["Quantity"].astype(int)
    transactions_df["Producer ID"] = transactions_df["Producer ID"].astype(int)

    return transactions_df


def save_transactions_to_excel(transactions_df):
    """Save the generated transactions to an Excel file with proper data types."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    start_date = transactions_df["Date"].min().strftime("%Y%m%d")
    end_date = transactions_df["Date"].max().strftime("%Y%m%d")
    filename = f"transactions_{start_date}_{end_date}.xlsx"
    filepath = os.path.join(SAVE_DIR, filename)

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        transactions_df.to_excel(writer, sheet_name="Transactions", index=False)

    print(f"Transaction data saved successfully to: {filepath}")


# ============================ MAIN EXECUTION ============================

if __name__ == "__main__":
    # Example of setting known prices for specific product names
    fixed_prices = {
        "Corn": 1.99,  # Del Monte Fresh Cut
        "Wheat": 12.99,  # Whole Wheat - Amazon
        "Soybeans": 18.00,  # 50 lb bag
        "Sugar": 4.89,  # Domino Sugar
        "Coffee": 6.99,  # Peet's Coffee
        "Beef": 10.99,  # Ground Beef
        "Milk": 3.24,  # Great Value Milk
        "Chocolate": 4.05,  # Tony's
    }

    demand_data = load_demand_data(DEMAND_FILEPATH)
    transactions_df = generate_transactions(demand_data, fixed_prices)
    save_transactions_to_excel(transactions_df)
