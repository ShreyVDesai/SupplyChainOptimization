import os
import pandas as pd
import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split

"""
This script performs data schema validation and statistics generation on a transaction dataset.
It uses TensorFlow Data Validation (TFDV) to:
1. Generate statistics about the data
2. Infer a schema from the data
3. Save the schema and statistics for future use in the data pipeline
"""

# Set up file paths
FILE_PATH = "transactions_20190103_20241231.xlsx"  # Source data file
SCHEMA_OUTPUT = "schema_output.json"  # Where the inferred schema will be saved
STATS_OUTPUT = "stats_output.json"  # Where the data statistics will be saved

# Check if file exists before proceeding
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Dataset file not found: {FILE_PATH}")

# Load dataset from Excel file into pandas DataFrame
print(f"Loading dataset from {FILE_PATH}...")
df = pd.read_excel(FILE_PATH)

# Split dataset into training (80%) and test (20%) sets
# shuffle=False maintains the original order of records (chronological for time series data)
print(f"Splitting dataset into training and test sets (80/20 split)...")
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Generate statistics for training and test datasets
# TFDV statistics include:
# - Feature types (numeric, categorical, etc.)
# - Data distributions
# - Missing value counts
# - Min/max values
# - Unique value counts for categorical features
print("Generating statistics for training and test datasets...")
train_stats = tfdv.generate_statistics_from_dataframe(train_df)
test_stats = tfdv.generate_statistics_from_dataframe(test_df)

# Infer schema from the training dataset
# A schema describes the expected properties of valid data, including:
# - Feature names and types
# - Value ranges
# - Expected distributions
# - Required vs. optional features
print("Inferring schema from training data...")
schema = tfdv.infer_schema(train_stats)

# Display the schema (useful when running interactively)
# This shows all features and their constraints
tfdv.display_schema(schema)


# Save schema and statistics as JSON files
# These files can be used to:
# - Validate new data against the schema
# - Monitor data drift over time
# - Document the dataset structure
# - Configure data preprocessing components
def save_json(data, filename):
    """
    Save TFDV schema or statistics objects to a JSON file.

    Args:
        data: A TFDV schema or statistics protocol buffer object
        filename: Destination file path for saving the output
    """
    with open(filename, "w") as f:
        f.write(str(data))


# Save the schema and training statistics to files
print(f"Saving schema to {SCHEMA_OUTPUT}...")
save_json(schema, SCHEMA_OUTPUT)
print(f"Saving training statistics to {STATS_OUTPUT}...")
save_json(train_stats, STATS_OUTPUT)

print(f"Schema and statistics saved: {SCHEMA_OUTPUT}, {STATS_OUTPUT}")
# The saved files can now be used in downstream data validation and processing steps

"""
What can you do with the schema and statistics:
1. Data Quality Checks: Validate new data against the schema to catch anomalies
2. Data Preprocessing: Configure feature transformations based on data types and ranges
3. Monitoring: Track changes in data distributions over time to detect data drift
4. Documentation: Use as documentation for the dataset structure and properties
"""
