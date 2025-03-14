#!/usr/bin/env bash
set -e

# Usage: ./scripts/deploy_airflow.sh [VM_NAME] [VM_ZONE] [REMOTE_USER]
# Defaults: VM_NAME=airflow-server, VM_ZONE=us-central1-a, REMOTE_USER=ubuntu
VM_NAME="${1:-${VM_NAME:-airflow-server}}"
VM_ZONE="${2:-${VM_ZONE:-us-central1-a}}"
REMOTE_USER="${3:-${REMOTE_USER:-ubuntu}}"


echo "🚀 Deploying Airflow on ${VM_NAME}..."

# Dynamically fetch the external IP of the VM.
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" --zone "$VM_ZONE" --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
echo "Fetched external IP: ${EXTERNAL_IP}"

# SSH into the VM and execute the deployment commands.
ssh -o StrictHostKeyChecking=no -i ~/.ssh/github-actions-key "${REMOTE_USER}@${EXTERNAL_IP}" <<EOF
  echo "🚀 Ensuring Docker is installed..."
  if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing..."
    sudo apt-get update -y
    echo "🚀 Adding Docker repository..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable"
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
  else
    echo "✅ Docker is already installed."
  fi

  if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Installing latest version..."
    DOCKER_COMPOSE_VERSION=\$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
    sudo curl -L "https://github.com/docker/compose/releases/download/\${DOCKER_COMPOSE_VERSION}/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
  else
    echo "✅ Docker Compose is already installed."
  fi

  # Give user Docker permissions.
  echo "🔄 Adding user to Docker group..."
  sudo usermod -aG docker \$USER
  newgrp docker
  sudo systemctl restart docker
  echo "✅ User added to Docker group and Docker restarted."

  # Fix Docker socket permissions.
  sudo chmod 666 /var/run/docker.sock
  echo "✅ Docker socket permissions fixed."

  mkdir -p /opt/airflow
  echo "airflow dir created."
  echo "🚀 Ensuring GCP Key File exists..."
  if [ -d /opt/airflow/gcp-key.json ]; then
      echo "⚠️ Found directory at /opt/airflow/gcp-key.json. Removing it..."
      sudo rm -rf /opt/airflow/gcp-key.json
  fi
  echo "🚀 Creating GCP Key File..."
  echo "$GCP_SERVICE_ACCOUNT_KEY" | jq . > /opt/airflow/gcp-key.json
  chmod 644 /opt/airflow/gcp-key.json
  sudo chown ubuntu:docker /opt/airflow/gcp-key.json
  echo "✅ GCP Key File Created."

  echo "🚀 Fixing Airflow log directory permissions..."
  sudo mkdir -p /opt/airflow/logs
  sudo chmod -R 777 /opt/airflow/logs
  sudo chown -R \$USER:\$USER /opt/airflow/logs
  
  cd /opt/airflow

  echo "🚀 Pulling the latest image from Artifact Registry..."
  gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
  docker compose pull || true

  echo "🚀 Stopping any running containers..."
  docker compose down || true

  # Remove postgres volume if you want to reset the DB (warning: this clears data)
  docker volume rm airflow_postgres-db-volume || true

  echo "🚀 Starting Airflow using Docker Compose..."
  docker compose up -d --remove-orphans

  echo "✅ Airflow successfully started!"
EOF
