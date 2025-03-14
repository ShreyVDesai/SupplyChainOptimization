#!/usr/bin/env bash
set -e

# Use environment variables (set in GitHub Actions or your local environment) or fallback defaults.
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-data-pipeline}"
IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
LOCATION="${GCP_LOCATION:-us-central1}"

FULL_IMAGE_PATH="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "🔍 Checking if '${IMAGE_NAME}:${IMAGE_TAG}' exists in Artifact Registry..."

# Get JSON output of images in Artifact Registry.
IMAGES_JSON=$(gcloud artifacts docker images list "${FULL_IMAGE_PATH}" --include-tags --format="json")

# Extract matching image name.
MATCHING_IMAGE=$(echo "$IMAGES_JSON" | jq -r --arg IMAGE_NAME "$FULL_IMAGE_PATH" '.[] | select(.package==$IMAGE_NAME) | .package')

# Initialize flag for build requirement.
BUILD_REQUIRED=false

if [[ "$MATCHING_IMAGE" == "$FULL_IMAGE_PATH" ]]; then
  echo "✅ Exact match found: '${IMAGE_NAME}:${IMAGE_TAG}'"
else
  echo "⚠️ No exact match for '${IMAGE_NAME}:${IMAGE_TAG}'. A new build is required."
  BUILD_REQUIRED=true
fi

echo "🔍 Checking if 'Dockerfile' or 'requirements.txt' has changed..."

if [ "$(git rev-list --count HEAD)" -lt 2 ]; then
  echo "⚠️ Not enough commit history. Assuming changes."
  BUILD_REQUIRED=true
else
  if ! git diff --quiet HEAD~1 HEAD -- Data_Pipeline/Dockerfile Data_Pipeline/requirements.txt; then
    echo "⚠️ Changes detected in 'Dockerfile' or 'requirements.txt'. A new build is required."
    BUILD_REQUIRED=true
  fi
fi

# Write the build requirement status to GitHub Actions output.
if [[ "$BUILD_REQUIRED" == "true" ]]; then
  echo "build_required=true" >> "$GITHUB_OUTPUT"
else
  echo "build_required=false" >> "$GITHUB_OUTPUT"
fi
