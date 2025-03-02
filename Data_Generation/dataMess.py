import random

import numpy as np
import pandas as pd


def mess_up_data(input_file, output_file):
    """Loads a dataset, sorts it by date first, introduces missing values, inconsistent formats, outliers, errors, duplicates, logical inconsistencies, and consecutive row deletions, then saves the modified file in a single sheet."""

    # Load the dataset manually without using chunksize
    df = pd.read_excel(input_file, dtype={"Date": str, "Quantity": object})

    # Convert Date column to datetime and sort first
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # df = df.sort_values(by='Date')  # Sort transactions by date first

    # Introduce missing values across the entire dataset (not per chunk)
    missing_values = {
        "Date": 0.01,
        "Cost Price": 0.01,
        "Quantity": 0.025,
        "Producer ID": 0.01,
        "Store Location": 0.02,
        "Product Name": 0.02,
    }

    for col, pct in missing_values.items():
        if col in df.columns:
            df.loc[df.sample(frac=pct).index, col] = (
                np.nan
            )  # Introduce missing values globally

    # Introduce inconsistent date formats
    if "Date" in df.columns:
        date_indices = df.sample(frac=0.02).index  # 2% inconsistent dates
        df.loc[date_indices, "Date"] = df.loc[date_indices, "Date"].apply(
            lambda x: (
                x.strftime("%d/%m/%Y")
                if pd.notna(x) and random.random() > 0.5
                else x.strftime("%m-%d-%Y") if pd.notna(x) else x
            )
        )

    # Split the dataset into smaller chunks manually
    chunk_size = 500000  # Process data in smaller subsets
    num_chunks = (len(df) // chunk_size) + 1  # Calculate number of chunks
    processed_chunks = []  # List to store processed chunks

    for i in range(num_chunks):
        chunk = df[i * chunk_size : (i + 1) * chunk_size]

        if chunk.empty:
            continue  # Skip empty chunks

        # Duplicate records
        duplicate_rows = chunk.sample(
            frac=0.01, random_state=42
        )  # 1% duplicate records
        chunk = pd.concat([chunk, duplicate_rows], ignore_index=True)

        if "Product Name" in chunk.columns:
            chunk.loc[chunk.sample(frac=0.03).index, "Product Name"] = (
                chunk["Product Name"]
                .astype(str)
                .apply(
                    lambda x: (
                        x.replace("a", "@")
                        if random.random() > 0.5
                        else x.replace("o", "0")
                    )
                )
            )  # 3% misspelled Product Names

        # Keep negative values for Cost Price but avoid excessive modifications
        if "Cost Price" in chunk.columns:
            chunk.loc[chunk.sample(frac=0.01).index, "Cost Price"] = -abs(
                chunk["Cost Price"].mean(skipna=True)
            )  # 1% negative cost prices

        # Introduce negative and zero values in Quantity column
        if "Quantity" in chunk.columns:
            chunk["Quantity"] = pd.to_numeric(
                chunk["Quantity"], errors="coerce"
            )  # Convert to numeric before applying transformations
            chunk.loc[chunk.sample(frac=0.02).index, "Quantity"] = -abs(
                chunk["Quantity"].mean(skipna=True)
            )  # 2% negative quantities
            chunk.loc[chunk.sample(frac=0.01).index, "Quantity"] = (
                0  # 1% zero quantities
            )

        # Introduce invalid store locations
        if "Store Location" in chunk.columns:
            chunk.loc[chunk.sample(frac=0.02).index, "Store Location"] = str(
                random.randint(1000, 9999)
            )  # 2% numeric store locations

        # Introduce logical inconsistencies
        if "Cost Price" in chunk.columns and "Quantity" in chunk.columns:
            inconsistent_indices = chunk.sample(
                frac=0.02
            ).index  # 1% logical inconsistencies
            chunk.loc[inconsistent_indices, "Cost Price"] = (
                0  # Cost Price is zero
            )
            chunk.loc[inconsistent_indices, "Quantity"] = random.randint(
                1, 10
            )  # Quantity is non-zero

        processed_chunks.append(chunk)  # Store the processed chunk

    # Merge all chunks into a single DataFrame
    final_df = pd.concat(processed_chunks, ignore_index=True)

    # Save entire dataset in a single sheet
    final_df.to_excel(
        output_file, sheet_name="Full_Data", index=False, engine="openpyxl"
    )

    print(f"Modified dataset saved as {output_file}")


# Example usage
input_file = (
    "transactions_20190103_20241231.xlsx"  # Replace with actual file path
)
output_file = "messy_transactions_20190103_20241231.xlsx"
mess_up_data(input_file, output_file)
