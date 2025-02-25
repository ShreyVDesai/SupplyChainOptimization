import os
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split
from tensorflow_metadata.proto.v0 import schema_pb2

# Set up file paths
FILE_PATH = "transactions_20190103_20241231.xlsx"
SCHEMA_OUTPUT = "schema_output.json"
STATS_OUTPUT = "stats_output.json"

# Check if file exists
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Dataset file not found: {FILE_PATH}")

# Load dataset
df = pd.read_excel(FILE_PATH)

# Split dataset
train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

# Generate statistics for training dataset
train_stats = tfdv.generate_statistics_from_dataframe(train_df)
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

# Infer schema
schema = tfdv.infer_schema(train_stats)
tfdv.display_schema(schema)

# Save schema and statistics
def save_json(data, filename):
    with open(filename, "w") as f:
        f.write(str(data))

save_json(schema, SCHEMA_OUTPUT)
save_json(train_stats, STATS_OUTPUT)

print(f"Schema and statistics saved: {SCHEMA_OUTPUT}, {STATS_OUTPUT}")
