import sys
from airflow import DAG
from airflow.providers.google.cloud.sensors.gcs import GCSObjectUpdateSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os

# Define default DAG arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),  # Start from yesterday
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define GCP parameters
BUCKET_NAME = "full-raw-data"
FILE_NAME = "data/new_data.csv"

# Define script paths
SCRIPTS = {
    "check": "DataValidation_Schema&Stats.py",
    "preprocess": "dataPreprocessing.py",
}


def run_script_in_container(script_name):
    """Run a script in the data-pipeline container"""
    try:
        import docker
    except ImportError:
        raise ImportError(
            "Docker package not installed. Please install 'docker' package in Airflow."
        )

    try:
        client = docker.from_env()
    except Exception as e:
        print(f"Error connecting to Docker: {e}")
        return "stop_pipeline"

    try:
        container = client.containers.get("data-pipeline-container")
    except docker.errors.NotFound:
        print("Data pipeline container not found. Make sure it's running.")
        return "stop_pipeline"
    except Exception as e:
        print(f"Error getting container: {e}")
        return "stop_pipeline"

    try:
        exit_code, output = container.exec_run(
            f"python {script_name}", workdir="/app/scripts"
        )
        print(f"Script output: {output.decode()}")

        if exit_code == 0:
            return "preprocess_data" if script_name == SCRIPTS["check"] else None
        else:
            print(f"Script failed with exit code {exit_code}")
            return "stop_pipeline"

    except Exception as e:
        print(f"Error running script in container: {e}")
        return "stop_pipeline"


# Dummy function to stop pipeline
def stop_pipeline():
    print("Data validation failed. Stopping pipeline.")


# Define DAG
with DAG(
    "data_pipeline_dag",
    default_args=default_args,
    description="Data pipeline that runs on GCP bucket upload",
    schedule_interval=None,
    catchup=False,
    tags=["data-pipeline"],
) as dag:

    # Wait for new file in GCP bucket
    wait_for_file = GCSObjectUpdateSensor(
        task_id="wait_for_file",
        bucket=BUCKET_NAME,
        object=FILE_NAME,
        google_cloud_conn_id="google_cloud_default",
        timeout=600,
        poke_interval=30,
        mode="poke",
    )

    # Run data checking with branching logic
    check_data = BranchPythonOperator(
        task_id="check_data",
        python_callable=run_script_in_container,
        op_args=[SCRIPTS["check"]],
    )

    # Run data preprocessing
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_script_in_container,
        op_args=[SCRIPTS["preprocess"]],
    )

    # Stop pipeline if data validation fails
    stop_task = PythonOperator(
        task_id="stop_pipeline", python_callable=stop_pipeline, trigger_rule="all_done"
    )

    # Define task dependencies
    wait_for_file >> check_data
    check_data >> [preprocess_data, stop_task]
