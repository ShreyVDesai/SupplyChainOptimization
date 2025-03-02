"""
Supply Chain Optimization - Scheduled Preprocessing DAG

This DAG runs automatically every minute to process files in the GCP bucket.
"""

from airflow import DAG
from supply_chain_common import DEFAULT_ARGS, create_preprocessing_tasks

# Create the scheduled DAG (runs every minute)
dag = DAG(
    "gcp_preprocessing_scheduled",
    default_args=DEFAULT_ARGS,
    description="Process files from GCP bucket using preprocessing.py (every minute)",
    schedule_interval="* * * * *",  # Run every minute
    catchup=False,
    tags=["supply-chain", "preprocessing", "scheduled"],
)

# Create tasks for the scheduled DAG
tasks = create_preprocessing_tasks(dag)
