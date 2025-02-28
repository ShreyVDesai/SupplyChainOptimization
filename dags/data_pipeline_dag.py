import sys
from airflow import DAG
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.providers.google.cloud.sensors.gcs import (
    GCSObjectsWithPrefixExistenceSensor,
)
from airflow.providers.google.cloud.transfers.gcs_to_local import (
    GCSToLocalFilesystemOperator,
)
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
import sys
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default DAG arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),  # Start from yesterday
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define GCP parameters
SOURCE_BUCKET_NAME = "full-raw-data"
DESTINATION_BUCKET_NAME = "fully-processed-data"  # New bucket for processed data
TEMP_LOCAL_PATH = "/tmp/airflow_data"
PROCESSED_PREFIX = "processed/"
OUTPUT_FORMAT = "json"  # Set output format to JSON

# Ensure temp directory exists
os.makedirs(TEMP_LOCAL_PATH, exist_ok=True)


# Helper function to dynamically import Python scripts
def import_script(script_path):
    """Import a Python script as a module"""
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_validation_script(**context):
    """Run the data validation script and determine next steps based on result"""
    # Get the downloaded file path from XCom
    ti = context["ti"]
    file_path = ti.xcom_pull(task_ids="download_new_file")

    if not file_path:
        logger.error("No file path received from previous task")
        return "stop_pipeline"

    logger.info(f"Running validation on file: {file_path}")

    try:
        # Dynamically load and run validation script
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts",
            "DataValidation_Schema&Stats.py",
        )

        # Import the script
        validation_module = import_script(script_path)

        # Run the analyze_data function
        has_issues = validation_module.analyze_data(file_path)

        if has_issues:
            logger.warning("Data validation detected issues. Stopping pipeline.")
            return "stop_pipeline"
        else:
            logger.info("Data validation passed. Proceeding to preprocessing.")
            return "preprocess_data"

    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return "stop_pipeline"


def run_preprocessing_script(**context):
    """Run the data preprocessing script"""
    # Get the downloaded file path from XCom
    ti = context["ti"]
    file_path = ti.xcom_pull(task_ids="download_new_file")
    filename = ti.xcom_pull(task_ids="identify_new_file", key="filename")

    if not file_path:
        logger.error("No file path received from previous task")
        return False

    logger.info(f"Running preprocessing on file: {file_path}")

    try:
        # Dynamically load and run preprocessing script
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts",
            "dataPreprocessing.py",
        )

        # Import the script
        preprocess_module = import_script(script_path)

        # Generate a base filename for the output
        base_filename = os.path.basename(filename)
        output_filename = f"processed_{base_filename}"

        # Change extension to json
        output_filename = f"{os.path.splitext(output_filename)[0]}.json"
        destination_path = f"cleaned_data/{output_filename}"

        # Run the main function with the new destination bucket
        output_file = os.path.join(TEMP_LOCAL_PATH, "cleaned_data.json")

        # Run preprocessing with explicit error capture
        try:
            logger.info(
                f"Calling preprocessing main function with destination: {DESTINATION_BUCKET_NAME}/{destination_path}"
            )
            success = preprocess_module.main(
                input_file=file_path,
                output_file=output_file,
                bucket_name=SOURCE_BUCKET_NAME,
                source_blob_name=filename,
                destination_bucket_name=DESTINATION_BUCKET_NAME,
                destination_blob_name=destination_path,
                output_format=OUTPUT_FORMAT,  # JSON output format
                cloud=True,
            )

            # Verify success and file existence
            if not success:
                logger.error("Preprocessing reported failure")
                return False

            # Check for local output file as a precaution
            if not os.path.exists(output_file):
                logger.warning(f"Local output file {output_file} was not created")

            logger.info(
                f"Data preprocessing completed successfully. Results should be in {DESTINATION_BUCKET_NAME}/{destination_path}"
            )

            # Store the destination path in XCom for potential future use or verification
            ti.xcom_push(
                key="output_destination",
                value=f"{DESTINATION_BUCKET_NAME}/{destination_path}",
            )
            return True

        except Exception as inner_e:
            logger.error(f"Error in preprocessing execution: {inner_e}", exc_info=True)
            return False

    except Exception as e:
        logger.error(f"Error during preprocessing setup: {e}", exc_info=True)
        return False


def verify_output_exists(**context):
    """Verify that the output file exists in the destination bucket"""
    ti = context["ti"]
    output_destination = ti.xcom_pull(
        task_ids="preprocess_data", key="output_destination"
    )

    if not output_destination:
        logger.error("No output destination received from preprocessing task")
        return False

    try:
        # Parse bucket and blob path
        parts = output_destination.split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid output destination format: {output_destination}")
            return False

        bucket_name, blob_name = parts

        # Import GCS libraries
        from google.cloud import storage

        # Check if the file exists
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            logger.info(f"Verified output file exists: {output_destination}")
            return True
        else:
            logger.error(f"Output file does not exist in GCS: {output_destination}")
            return False

    except Exception as e:
        logger.error(f"Error verifying output file: {e}", exc_info=True)
        return False


# Define DAG
with DAG(
    "data_pipeline_dag",
    default_args=default_args,
    description="Data pipeline that runs on any new file in GCP bucket, outputs JSON",
    schedule_interval=None,
    catchup=False,
    tags=["data-pipeline"],
) as dag:

    # Check for new files in GCS bucket (excluding processed files)
    check_for_new_files = GCSListObjectsOperator(
        task_id="check_for_new_files",
        bucket=SOURCE_BUCKET_NAME,
        prefix="",
        delimiter="",
        gcp_conn_id="google_cloud_default",
        do_xcom_push=True,
    )

    # Filter files to find unprocessed ones
    def find_new_file(**context):
        """Find unprocessed files from the GCS list"""
        ti = context["ti"]
        file_list = ti.xcom_pull(task_ids="check_for_new_files")

        if not file_list:
            logger.info("No files found in bucket")
            return None

        # Filter out processed files and directories
        unprocessed_files = [
            f
            for f in file_list
            if not f.startswith(PROCESSED_PREFIX) and not f.endswith("/")
        ]

        if not unprocessed_files:
            logger.info("No unprocessed files found")
            return None

        # Take the first unprocessed file
        file_to_process = unprocessed_files[0]
        logger.info(f"Found file to process: {file_to_process}")

        # Store the filename in XCom
        context["ti"].xcom_push(key="filename", value=file_to_process)

        return file_to_process

    # Task to identify a new file
    identify_new_file = PythonOperator(
        task_id="identify_new_file",
        python_callable=find_new_file,
        provide_context=True,
        trigger_rule="all_success",
    )

    # Wait for the identified file to exist (as a sensor)
    def wait_for_file(**context):
        ti = context["ti"]
        filename = ti.xcom_pull(task_ids="identify_new_file")

        if filename:
            # Create a sensor task dynamically
            sensor = GCSObjectsWithPrefixExistenceSensor(
                task_id="wait_for_file",
                bucket=SOURCE_BUCKET_NAME,
                prefix=filename,
                google_cloud_conn_id="google_cloud_default",
                poke_interval=30,
                timeout=600,
                dag=dag,
            )
            return sensor.execute(context)
        else:
            logger.info("No file to wait for")
            return False

    wait_for_file_task = PythonOperator(
        task_id="wait_for_file_task",
        python_callable=wait_for_file,
        provide_context=True,
        trigger_rule="all_success",
    )

    # Download the new file to local filesystem
    def download_file(**context):
        ti = context["ti"]
        filename = ti.xcom_pull(task_ids="identify_new_file")

        if not filename:
            logger.warning("No filename provided for download")
            return None

        local_path = os.path.join(TEMP_LOCAL_PATH, os.path.basename(filename))

        # Create a download task dynamically
        download_task = GCSToLocalFilesystemOperator(
            task_id="download_file",
            object_name=filename,
            bucket=SOURCE_BUCKET_NAME,
            filename=local_path,
            gcp_conn_id="google_cloud_default",
            dag=dag,
        )
        download_task.execute(context)

        return local_path

    download_new_file = PythonOperator(
        task_id="download_new_file",
        python_callable=download_file,
        provide_context=True,
        trigger_rule="all_success",
    )

    # Branch based on validation result
    check_data = BranchPythonOperator(
        task_id="check_data",
        python_callable=run_validation_script,
        provide_context=True,
    )

    # Preprocessing task
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing_script,
        provide_context=True,
        trigger_rule="none_failed_or_skipped",
    )

    # Verification task to check if output file exists in GCS
    verify_output = PythonOperator(
        task_id="verify_output",
        python_callable=verify_output_exists,
        provide_context=True,
        trigger_rule="all_success",
    )

    # Stop pipeline if validation fails - this is a terminal task
    stop_pipeline = DummyOperator(
        task_id="stop_pipeline", trigger_rule="none_failed_or_skipped"
    )

    # Successful completion
    pipeline_complete = DummyOperator(
        task_id="pipeline_complete", trigger_rule="none_failed_min_one_success"
    )

    # Pipeline failure
    pipeline_failed = DummyOperator(
        task_id="pipeline_failed", trigger_rule="one_failed"
    )

    # Define task dependencies with clear branching logic
    (
        check_for_new_files
        >> identify_new_file
        >> wait_for_file_task
        >> download_new_file
        >> check_data
    )

    # Branching paths after check_data
    check_data >> [preprocess_data, stop_pipeline]

    # Add verification step before marking pipeline complete
    preprocess_data >> verify_output >> pipeline_complete

    # If verification fails, go to pipeline_failed
    verify_output.set_downstream(pipeline_failed)

    # Error handling - connect the failed states
    [
        check_for_new_files,
        identify_new_file,
        wait_for_file_task,
        download_new_file,
        check_data,
        preprocess_data,
    ] >> pipeline_failed
