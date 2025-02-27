from airflow import DAG
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import os
import subprocess

# Define default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 21),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define GCP parameters
BUCKET_NAME = "full-raw-data"
FILE_NAME = "data/new_data.csv"

# Define script paths
SCRIPT_DIR = "/Data-Pipeline/scripts"
SCRIPTS = {
    "check": "DataValidation_Schema&Stats.py",
    "preprocess": "dataPreprocessing.py"
}

# Function to execute Python scripts
def run_script(script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    try:
        subprocess.run(["python3", script_path], check=True)
        return "preprocess_data"  # If successful, continue to preprocessing
    except subprocess.CalledProcessError:
        return "stop_pipeline"  # If failed, stop execution

# Dummy function to stop pipeline
def stop_pipeline():
    print("Data validation failed. Stopping pipeline.")

# Define DAG
with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Data pipeline that runs on GCP bucket upload',
    schedule_interval=None,
    catchup=False
) as dag:

    # Wait for new file in GCP bucket
    wait_for_file = GCSObjectExistenceSensor(
        task_id="wait_for_file",
        bucket=BUCKET_NAME,
        object=FILE_NAME,
        google_cloud_conn_id="google_cloud_default",
        timeout=600,
        poke_interval=30,
    )

    # Run data checking with branching logic
    check_data = BranchPythonOperator(
        task_id="check_data",
        python_callable=run_script,
        op_args=[SCRIPTS["check"]],
    )

    # Run data preprocessing
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_script,
        op_args=[SCRIPTS["preprocess"]],
    )

    # Stop pipeline if data validation fails
    stop_task = PythonOperator(
        task_id="stop_pipeline",
        python_callable=stop_pipeline
    )

    # Define task dependencies
    wait_for_file >> check_data
    check_data >> [preprocess_data, stop_task]
