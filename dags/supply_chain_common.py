"""
Supply Chain Optimization - Common Module

This module contains shared functions and parameters used by the preprocessing DAGs.
"""

from datetime import datetime, timedelta
import docker
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook

# Define default arguments
DEFAULT_ARGS = {
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
PROCESSED_CACHE_BUCKET_NAME = "fully-processed-cache"
DVC_REMOTE_NAME = "fully-processed-data-dvc"


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


def run_pre_validation(**context):
    """Run the pre_validation script to validate data before processing"""
    client = docker.from_env()

    try:
        # Get bucket name from xcom
        gcs_bucket = (
            context["ti"].xcom_pull(
                task_ids="print_gcs_info", key="gcs_bucket"
            )
            or GCP_BUCKET_NAME
        )

        container = client.containers.get("data-pipeline-container")
        print(f"Container found: {container.name}")

        # Execute the pre_validation.py script with bucket parameter
        exit_code, output = container.exec_run(
            cmd=f"python pre_validation.py --bucket={gcs_bucket}",
            workdir="/app/scripts",
        )

        output_str = output.decode("utf-8")
        print(f"Pre-validation output: {output_str}")

        # Check if there are any files left in the bucket after validation
        hook = GCSHook(gcp_conn_id=GCP_CONNECTION_ID)
        remaining_files = hook.list(bucket_name=gcs_bucket)

        if not remaining_files:
            print("No valid files remain in the bucket after pre-validation.")
            raise AirflowSkipException("No valid files to process")

        # Interpret exit code from pre_validation.py:
        # 0 = All files valid
        # 1 = Some files valid, some invalid (removed)
        # 2 = Critical failure or all files invalid
        if exit_code == 0:
            print("Pre-validation completed successfully for all files")
            context["ti"].xcom_push(key="validation_status", value="full")
        elif exit_code == 1:
            print(
                "Some files failed validation, but continuing with valid ones"
            )
            context["ti"].xcom_push(key="validation_status", value="partial")
        elif exit_code == 2:
            # If there are still files in the bucket, there must be valid files
            # This handles the case where pre_validation.py miscategorized some
            # files
            if remaining_files:
                print(
                    "Partial validation - continuing with remaining files in bucket"
                )
                context["ti"].xcom_push(
                    key="validation_status", value="partial"
                )
            else:
                print("No valid files to process after pre-validation")
                raise AirflowSkipException(
                    "No valid files after pre-validation"
                )
        else:
            # Unknown exit code, check remaining files to decide
            if remaining_files:
                print(
                    f"Unknown validation status (code {exit_code}), but files remain. Continuing."
                )
                context["ti"].xcom_push(
                    key="validation_status", value="unknown"
                )
            else:
                print(
                    f"Unknown validation status (code {exit_code}) and no files remain."
                )
                raise AirflowSkipException(
                    "Unknown validation status and no files remain"
                )

        return output_str

    except docker.errors.NotFound:
        error_msg = (
            "data-pipeline-container not found. Make sure it's running."
        )
        print(error_msg)
        raise Exception(error_msg)
    except AirflowSkipException:
        # Re-raise AirflowSkipException to ensure downstream tasks are skipped
        raise
    except Exception as e:
        print(f"Error running pre-validation script: {str(e)}")
        raise


def run_preprocessing_script(**context):
    """Run the preprocessing script in the existing data-pipeline-container"""
    client = docker.from_env()

    try:
        # Get bucket name from xcom
        gcs_bucket = (
            context["ti"].xcom_pull(
                task_ids="print_gcs_info", key="gcs_bucket"
            )
            or GCP_BUCKET_NAME
        )

        container = client.containers.get("data-pipeline-container")
        print(f"Container found: {container.name}")

        # Execute the preprocessing.py script with bucket parameters, including cache bucket
        exit_code, output = container.exec_run(
            cmd=f"python preprocessing.py --source_bucket={gcs_bucket} --destination_bucket={PROCESSED_BUCKET_NAME} --cache_bucket={PROCESSED_CACHE_BUCKET_NAME} --delete_after",
            workdir="/app/scripts",
        )

        output_str = output.decode("utf-8")
        print(f"Script output: {output_str}")

        if exit_code != 0:
            print(f"Script failed with exit code: {exit_code}")
            raise Exception(
                f"preprocessing.py failed with exit code {exit_code}"
            )

        print("Preprocessing completed successfully")
        return output_str

    except docker.errors.NotFound:
        error_msg = (
            "data-pipeline-container not found. Make sure it's running."
        )
        print(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        print(f"Error running preprocessing script: {str(e)}")
        raise


def run_dvc_versioning(**context):
    """Run the DVC versioning script to track processed data"""
    client = docker.from_env()

    try:
        # Get bucket name from xcom - no longer needed since we removed source_bucket parameter
        # Only kept for logging purposes
        gcs_bucket = (
            context["ti"].xcom_pull(
                task_ids="print_gcs_info", key="gcs_bucket"
            )
            or GCP_BUCKET_NAME
        )

        print(f"Source bucket (for reference only): {gcs_bucket}")

        container = client.containers.get("data-pipeline-container")
        print(f"Container found: {container.name}")

        # Execute the dvc_versioning.py script with bucket parameters
        # Removed source_bucket parameter as it's not used by the script
        exit_code, output = container.exec_run(
            cmd=f"python dvc_versioning.py --cache_bucket={PROCESSED_CACHE_BUCKET_NAME} --dvc_remote={DVC_REMOTE_NAME}",
            workdir="/app/scripts",
        )

        output_str = output.decode("utf-8")
        print(f"DVC versioning output: {output_str}")

        if exit_code != 0:
            print(f"DVC versioning failed with exit code: {exit_code}")
            raise Exception(
                f"dvc_versioning.py failed with exit code {exit_code}"
            )

        print("DVC versioning completed successfully")
        return output_str

    except docker.errors.NotFound:
        error_msg = (
            "data-pipeline-container not found. Make sure it's running."
        )
        print(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        print(f"Error running DVC versioning script: {str(e)}")
        raise


def create_preprocessing_tasks(dag):
    """Create all preprocessing tasks for a given DAG"""

    # Print information about the GCS event
    print_gcs_info_task = PythonOperator(
        task_id="print_gcs_info",
        python_callable=print_gcs_info,
        dag=dag,
    )

    # Get list of files in the bucket
    get_file_list_task = PythonOperator(
        task_id="get_file_list",
        python_callable=list_new_files,
        dag=dag,
    )

    # Run pre-validation to check data quality
    run_pre_validation_task = PythonOperator(
        task_id="run_pre_validation",
        python_callable=run_pre_validation,
        dag=dag,
    )

    # Run the preprocessing script in the existing data-pipeline-container
    run_preprocessing_task = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocessing_script,
        dag=dag,
    )

    # Run DVC versioning to track processed data
    run_dvc_versioning_task = PythonOperator(
        task_id="run_dvc_versioning",
        python_callable=run_dvc_versioning,
        dag=dag,
    )

    # Define the task dependencies
    (
        print_gcs_info_task
        >> get_file_list_task
        >> run_pre_validation_task
        >> run_preprocessing_task
        >> run_dvc_versioning_task
    )

    return {
        "print_gcs_info": print_gcs_info_task,
        "get_file_list": get_file_list_task,
        "run_pre_validation": run_pre_validation_task,
        "run_preprocessing": run_preprocessing_task,
        "run_dvc_versioning": run_dvc_versioning_task,
    }
