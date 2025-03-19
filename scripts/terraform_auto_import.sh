#!/bin/bash
set -e

# Change into the terraform configuration directory
cd terraform

# Use environment variables provided by the workflow.
PROJECT_ID="${TF_VAR_project_id}"
REGION="${TF_VAR_region}"
ARTIFACT_REPO="${ARTIFACT_REGISTRY_NAME}"
VM_NAME="${VM_NAME}"        # Provided by your workflow environment
VM_ZONE="${VM_ZONE}"        # Provided by your workflow environment
FIREWALL_NAME="allow-airflow-server"

echo "Checking if Artifact Registry exists..."
artifact_exists=$(gcloud artifacts repositories list \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --filter="name:projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}" \
  --format="value(name)")

if [ -n "$artifact_exists" ]; then
  echo "Artifact Registry exists. Importing it into Terraform..."
  terraform import google_artifact_registry_repository.airflow_docker "projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}"
else
  echo "Artifact Registry does not exist. Terraform will create it."
fi

echo "Checking if Firewall exists..."
firewall_exists=$(gcloud compute firewall-rules list \
  --project="${PROJECT_ID}" \
  --filter="name=${FIREWALL_NAME}" \
  --format="value(name)")

if [ -n "$firewall_exists" ]; then
  echo "Firewall exists. Importing it into Terraform..."
  terraform import google_compute_firewall.airflow_server "projects/${PROJECT_ID}/global/firewalls/${FIREWALL_NAME}"
else
  echo "Firewall does not exist. Terraform will create it."
fi

echo "Checking if Compute Instance exists..."
instance_exists=$(gcloud compute instances list \
  --project="${PROJECT_ID}" \
  --zones="${VM_ZONE}" \
  --filter="name=${VM_NAME}" \
  --format="value(name)")

if [ -n "$instance_exists" ]; then
  echo "Compute Instance exists. Importing it into Terraform..."
  terraform import google_compute_instance.airflow_server "projects/${PROJECT_ID}/zones/${VM_ZONE}/instances/${VM_NAME}"
else
  echo "Compute Instance does not exist. Terraform will create it."
fi

echo "Running Terraform plan and apply..."
terraform plan -out=tfplan
terraform apply -auto-approve tfplan

echo "Waiting for VM to fully initialize..."
sleep 60
