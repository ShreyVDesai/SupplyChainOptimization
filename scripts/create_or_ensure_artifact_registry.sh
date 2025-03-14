#!/usr/bin/env bash
set -e

# --------- Configuration -----------
REGISTRY_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
LOCATION="${GCP_LOCATION:-us-central1}"
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_FORMAT="${REPO_FORMAT:-docker}"

echo "🔍 Checking if Artifact Registry '$REGISTRY_NAME' exists in '$LOCATION' for project '$PROJECT_ID'..."

# Use the full resource name filter
EXISTING_REPO=$(
  gcloud artifacts repositories list \
    --project="$PROJECT_ID" \
    --location="$LOCATION" \
    --filter="name=projects/$PROJECT_ID/locations/$LOCATION/repositories/$REGISTRY_NAME" \
    --format="value(name)"
)

if [[ -z "$EXISTING_REPO" ]]; then
  echo "❌ Repository '$REGISTRY_NAME' not found. Creating it..."
  gcloud artifacts repositories create "$REGISTRY_NAME" \
    --project="$PROJECT_ID" \
    --repository-format="$REPO_FORMAT" \
    --location="$LOCATION" \
    --description="Docker repository for data-pipeline"
  echo "✅ Artifact Registry '$REGISTRY_NAME' created."
else
  echo "✅ Artifact Registry '$REGISTRY_NAME' already exists: $EXISTING_REPO"
fi
