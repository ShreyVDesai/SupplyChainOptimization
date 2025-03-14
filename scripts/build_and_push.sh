#!/usr/bin/env bash
set -e

# Use environment variables or fallback defaults.
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-data-pipeline}"
# Generate a new image tag based on the current date and time.
IMAGE_TAG=$(date +'%Y%m%d-%H%M%S')
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
REPO_URI="${GCP_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo "🚀 Docker build triggered (image missing or code changed)."
echo "Building image: ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the Docker image with --no-cache to ensure a fresh build.
docker build --no-cache -t "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" -f Data_Pipeline/Dockerfile Data_Pipeline

# Also tag this build as 'latest' for convenience.
docker tag "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" "${REPO_URI}/${IMAGE_NAME}:latest"

# Push both the timestamped tag and the 'latest' tag.
docker push "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REPO_URI}/${IMAGE_NAME}:latest"

echo "Image pushed successfully with tag: ${IMAGE_TAG}"
