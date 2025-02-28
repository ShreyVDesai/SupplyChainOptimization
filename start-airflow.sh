#!/bin/bash

# Set working directory to the script's directory
cd "$(dirname "$0")"

# Check if secret/.env exists
if [ ! -f "./secret/.env" ]; then
  echo "Error: secret/.env file not found!"
  echo "Please make sure you have created a .env file in the secret directory."
  exit 1
fi

# Load environment variables from secret/.env
export $(grep -v '^#' ./secret/.env | xargs)

# Additional check for GCP key file
if [ ! -f "./secret/gcp-key.json" ]; then
  echo "Warning: GCP key file (secret/gcp-key.json) not found!"
  echo "Make sure your GCP credentials are properly configured."
fi

# Note about .env file
echo "Note: You can now also use 'docker compose up' directly since we've created a .env file."
echo "      The script is still useful for loading any custom values from secret/.env."

# Start the Docker Compose services
docker-compose up -d

echo "Airflow services starting. Web UI will be available at http://localhost:8080"
echo "Username: ${AIRFLOW_ADMIN_USERNAME:-admin}"
echo "Password: ${AIRFLOW_ADMIN_PASSWORD:-admin}" 