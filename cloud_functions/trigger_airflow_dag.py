import base64
import json
import os
import requests
from google.cloud import storage
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import functions_framework

# Airflow webserver details
AIRFLOW_WEBSERVER_URL = os.environ.get("AIRFLOW_WEBSERVER_URL", "http://localhost:8080")
DAG_ID = "gcp_preprocessing_dag"
IAP_CLIENT_ID = os.environ.get("IAP_CLIENT_ID", "")  # Only needed if using IAP


@functions_framework.cloud_event
def trigger_airflow_dag(cloud_event):
    """
    Cloud Function triggered by GCS event to trigger Airflow DAG.

    Args:
        cloud_event: The Cloud Event that triggered this function
    """
    # Extract bucket and file info from the event
    data = cloud_event.data

    print(
        f"Function triggered by upload of: {data['name']} to bucket: {data['bucket']}"
    )

    # Trigger the Airflow DAG
    return trigger_dag(DAG_ID, data)


def trigger_dag(dag_id, gcs_data):
    """
    Trigger an Airflow DAG with GCS event data.

    Args:
        dag_id: ID of the DAG to trigger
        gcs_data: GCS event data containing bucket and file information
    """
    # The endpoint for triggering a DAG
    endpoint = f"{AIRFLOW_WEBSERVER_URL}/api/v1/dags/{dag_id}/dagRuns"

    # Set headers based on auth method
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # If using Basic Auth (local development)
    if "AIRFLOW_USERNAME" in os.environ and "AIRFLOW_PASSWORD" in os.environ:
        import base64

        auth_string = (
            f"{os.environ['AIRFLOW_USERNAME']}:{os.environ['AIRFLOW_PASSWORD']}"
        )
        encoded_auth = base64.b64encode(auth_string.encode()).decode("utf-8")
        headers["Authorization"] = f"Basic {encoded_auth}"

    # If using IAP (production)
    elif IAP_CLIENT_ID:
        try:
            open_id_connect_token = id_token.fetch_id_token(Request(), IAP_CLIENT_ID)
            headers["Authorization"] = f"Bearer {open_id_connect_token}"
        except Exception as e:
            print(f"Error getting ID token: {e}")
            return f"Error getting ID token: {e}", 500

    # Prepare the payload for the Airflow API
    payload = {
        "conf": {
            "gcs_bucket": gcs_data["bucket"],
            "gcs_object": gcs_data["name"],
            "event_time": gcs_data.get("timeCreated", ""),
        }
    }

    # Make the request to Airflow API
    try:
        response = requests.post(endpoint, headers=headers, json=payload)

        if response.status_code == 200 or response.status_code == 201:
            print(f"Successfully triggered DAG {dag_id}: {response.text}")
            return f"DAG {dag_id} triggered successfully", 200
        else:
            print(f"Error triggering DAG: {response.status_code} - {response.text}")
            return (
                f"Error triggering DAG: {response.status_code} - {response.text}",
                500,
            )

    except Exception as e:
        print(f"Exception triggering DAG: {str(e)}")
        return f"Exception triggering DAG: {str(e)}", 500
