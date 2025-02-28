#!/bin/bash

# Set working directory to the script's directory
cd "$(dirname "$0")"

echo "===== Supply Chain Optimization Setup ====="
echo "Starting Airflow services with production configuration..."

# Check if secret/.env exists
if [ ! -f "./secret/.env" ]; then
  echo "Error: secret/.env file not found!"
  echo "Please make sure you have created a .env file in the secret directory."
  exit 1
fi

echo "Loading environment variables from secret/.env..."
# Load environment variables from secret/.env (these will override the .env in root)
export $(grep -v '^#' ./secret/.env | xargs)

# Additional check for GCP key file
if [ ! -f "./secret/gcp-key.json" ]; then
  echo "Warning: GCP key file (secret/gcp-key.json) not found!"
  echo "Make sure your GCP credentials are properly configured."
fi

echo "Note: This script loads variables from secret/.env which override those in the root .env file."
echo "      For simple development, you can also use 'docker compose up' directly."

# Start the Docker Compose services
docker-compose up -d

echo ""
echo "Airflow services starting. Web UI will be available at http://localhost:8080"
echo "Username: ${AIRFLOW_ADMIN_USERNAME:-admin}"
echo "Password: ${AIRFLOW_ADMIN_PASSWORD:-admin}"
echo "============================================" 