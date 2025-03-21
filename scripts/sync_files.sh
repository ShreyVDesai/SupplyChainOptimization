#!/usr/bin/env bash
set -e

VM_NAME="${VM_NAME:-airflow-server}"
VM_ZONE="${VM_ZONE:-us-central1-a}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"

echo "🚀 Syncing project files to $VM_NAME..."

# Dynamically fetch the external IP of the VM
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" --zone "$VM_ZONE" --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
echo "Fetched external IP: $EXTERNAL_IP"

# Re-add SSH public key metadata (in case the VM reboot wiped it)
echo "🔐 Updating VM SSH metadata with public key..."
PUBKEY=$(cat ~/.ssh/github-actions-key.pub)
gcloud compute instances add-metadata "$VM_NAME" \
  --zone="$VM_ZONE" \
  --metadata="ssh-keys=${REMOTE_USER}:${PUBKEY}"

# Wait until SSH (port 22) is available
echo "Waiting for SSH service to be available on port 22..."
until nc -z "$EXTERNAL_IP" 22; do
  echo "SSH not available yet, waiting 5 seconds..."
  sleep 5
done
echo "SSH is now available on $EXTERNAL_IP:22"

# Prepare remote directory
ssh -o StrictHostKeyChecking=no -t -i ~/.ssh/github-actions-key "${REMOTE_USER}@${EXTERNAL_IP}" <<EOF
  sudo mkdir -p /opt/airflow
  sudo chown -R ${REMOTE_USER}:${REMOTE_USER} /opt/airflow
  sudo chmod -R 775 /opt/airflow
EOF

# Sync project files using rsync
rsync -avz -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/github-actions-key" --exclude '.git' . "${REMOTE_USER}@${EXTERNAL_IP}":/opt/airflow

echo "✅ Files synced successfully."
