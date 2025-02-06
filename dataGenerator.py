import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_synthetic_data(start_date, end_date, num_products):
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize dataframe
    df = pd.DataFrame(index=dates)

    # Simulate demand for each product
    for product_id in range(1, num_products + 1):
        # Define base demand (e.g., average demand)
        base_demand = 100 + product_id * 10  # Base demand per product
        # Define seasonality (e.g., yearly cycle, holidays, weekends)
        seasonality = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)  # Simple yearly seasonality
        # Introduce trend (optional)
        trend = np.linspace(0, 50, len(dates))  # Increasing demand over time
        # Introduce noise (e.g., random variation)
        noise = np.random.normal(0, 5, len(dates))  # Gaussian noise

        # Simulate the product demand
        product_demand = base_demand + seasonality + trend + noise
        df[f'Product_{product_id}'] = product_demand

    return df

def visualize_data(df):
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title('Synthetic Demand for Retail Products')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_to_excel(df, start_date, end_date):
    # Define the directory
    save_dir = "./.data/"
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Define the filename
    filename = f"{start_date}_{end_date}.xlsx"
    filepath = os.path.join(save_dir, filename)

    # Save to Excel
    df.to_excel(filepath)
    print(f"Data saved successfully to: {filepath}")

# Define the start and end dates
start_date = '2024-01-01'
end_date = '2024-12-31'
num_products = 5  # Number of products

# Generate the synthetic data
synthetic_data = generate_synthetic_data(start_date, end_date, num_products)

# Visualize the data before saving
visualize_data(synthetic_data)

# Save the data to an Excel file
save_to_excel(synthetic_data, start_date, end_date)
