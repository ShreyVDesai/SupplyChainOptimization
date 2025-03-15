#!/usr/bin/env bash
set -e

# Usage: ./scripts/configure_ssh.sh <VM_NAME> <VM_ZONE> <REMOTE_USER>
VM_NAME="${VM_NAME:-airflow-server}"
VM_ZONE="${VM_ZONE:-us-central1-a}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"

echo "🚀 Ensuring SSH key for GitHub Actions..."
if [ ! -f ~/.ssh/github-actions-key ]; then
  ssh-keygen -t rsa -b 4096 -C "github-actions" -N "" -f ~/.ssh/github-actions-key
  echo "✅ SSH Key generated!"
else
  echo "✅ SSH Key already exists!"
fi

PUBLIC_KEY=$(cat ~/.ssh/github-actions-key.pub)

echo "🔑 Attaching SSH key to VM metadata for $VM_NAME in zone $VM_ZONE..."
EXISTING_KEYS=$(gcloud compute instances describe "$VM_NAME" --zone "$VM_ZONE" --format="value(metadata.ssh-keys)" 2>/dev/null || echo "")
if [[ "$EXISTING_KEYS" != *"$PUBLIC_KEY"* ]]; then
  gcloud compute instances add-metadata "$VM_NAME" --zone "$VM_ZONE" \
    --metadata=ssh-keys="${REMOTE_USER}:$PUBLIC_KEY"
  echo "✅ SSH key added to VM!"
else
  echo "✅ SSH key is already in VM metadata!"
fi
