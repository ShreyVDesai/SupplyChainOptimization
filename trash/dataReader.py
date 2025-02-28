import pandas as pd
import matplotlib.pyplot as plt
import os


def read_excel(filepath):
    """Read the synthetic demand data from an Excel file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    df = pd.read_excel(filepath, sheet_name="Sheet1", index_col=0, parse_dates=True)
    return df


def visualize_data(df):
    """Plot the demand data and save it as an image."""
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column, alpha=0.7)

    plt.title("Visualized Synthetic Demand Data")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    # Save the plot instead of showing it
    output_path = "/data/demand_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Define the expected file path (modify as needed)
    filepath = "/data/commodity_demand_20190103_20241231.xlsx"

    try:
        data = read_excel(filepath)
        visualize_data(data)
    except FileNotFoundError as e:
        print(e)
