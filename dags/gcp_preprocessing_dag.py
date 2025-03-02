"""
Supply Chain Optimization - GCP Preprocessing DAG

This DAG is triggered when a new file is uploaded to the 'full-raw-data' GCP bucket.
It then runs the preprocessing.py script in the existing data-pipeline-container.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.models.param import Param

# Define default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 3, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define GCP bucket parameters
GCP_BUCKET_NAME = "full-raw-data"
GCP_CONNECTION_ID = "google_cloud_default"
PROCESSED_BUCKET_NAME = "fully-processed-data"


def print_gcs_info(**context):
    """Print information about the GCS event that triggered the DAG"""
    dag_run = context["dag_run"]
    print(f"DAG triggered by event: {dag_run.conf}")

    # Extract parameters from the event
    gcs_bucket = dag_run.conf.get("gcs_bucket", GCP_BUCKET_NAME)
    gcs_object = dag_run.conf.get("gcs_object", "")
    event_time = dag_run.conf.get("event_time", "")

    print(f"GCS Event Details:")
    print(f"  Bucket: {gcs_bucket}")
    print(f"  Object: {gcs_object}")
    print(f"  Event Time: {event_time}")

    # Store for downstream tasks
    context["ti"].xcom_push(key="gcs_bucket", value=gcs_bucket)
    context["ti"].xcom_push(key="gcs_object", value=gcs_object)

    return {
        "gcs_bucket": gcs_bucket,
        "gcs_object": gcs_object,
        "event_time": event_time,
    }


def list_new_files(**context):
    """List all files in the bucket and return their names"""
    # Use the bucket from the GCS event if available
    gcs_bucket = (
        context["ti"].xcom_pull(task_ids="print_gcs_info", key="gcs_bucket")
        or GCP_BUCKET_NAME
    )

    hook = GCSHook(gcp_conn_id=GCP_CONNECTION_ID)
    files = hook.list(bucket_name=gcs_bucket)
    context["ti"].xcom_push(key="file_list", value=files)
    print(f"Found {len(files)} files in bucket: {files}")
    return files


def run_preprocessing_script(**context):
    """Run the preprocessing script in the existing data-pipeline-container"""
    import docker

    client = docker.from_env()

    try:
        container = client.containers.get("data-pipeline-container")
        print(f"Container found: {container.name}")

        # Execute the preprocessing.py script
        exit_code, output = container.exec_run(
            cmd="python preprocessing.py", workdir="/app/scripts"
        )

        output_str = output.decode("utf-8")
        print(f"Script output: {output_str}")

        if exit_code != 0:
            print(f"Script failed with exit code: {exit_code}")
            raise Exception(f"preprocessing.py failed with exit code {exit_code}")

        print("Preprocessing completed successfully")
        return output_str

    except docker.errors.NotFound:
        error_msg = "data-pipeline-container not found. Make sure it's running."
        print(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        print(f"Error running preprocessing script: {str(e)}")
        raise


with DAG(
    "gcp_preprocessing_dag",
    default_args=default_args,
    description="Process files from GCP bucket using preprocessing.py",
    schedule_interval=None,  # External trigger only
    catchup=False,
    tags=["supply-chain", "preprocessing"],
) as dag:

    # Print information about the GCS event
    print_gcs_info = PythonOperator(
        task_id="print_gcs_info",
        python_callable=print_gcs_info,
    )

    # Get list of files in the bucket
    get_file_list = PythonOperator(
        task_id="get_file_list",
        python_callable=list_new_files,
    )

    # Run the preprocessing script in the existing data-pipeline-container
    run_preprocessing = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocessing_script,
    )

    # Define the task dependencies
    print_gcs_info >> get_file_list >> run_preprocessing
