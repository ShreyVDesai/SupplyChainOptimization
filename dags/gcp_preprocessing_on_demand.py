"""
Supply Chain Optimization - On-Demand Preprocessing DAG

This DAG is designed to be triggered manually for on-demand processing of files in GCP bucket.
"""

from airflow import DAG
from supply_chain_common import DEFAULT_ARGS, create_preprocessing_tasks

# Create the on-demand DAG (manually triggered)
dag = DAG(
    "gcp_preprocessing_on_demand",
    default_args=DEFAULT_ARGS,
    description="Process files from GCP bucket using preprocessing.py (on-demand)",
    schedule_interval=None,  # Only triggered manually
    catchup=False,
    tags=["supply-chain", "preprocessing", "on-demand"],
)

# Create tasks for the on-demand DAG
tasks = create_preprocessing_tasks(dag)
