#!/usr/bin/env bash
set -e 

# ----------- CONFIGURATION -----------
VM_NAME="airflow-server"
VM_ZONE="us-central1-a"

# Example ARM-based machine type in GCP (4 vCPU, 16GB)
MACHINE_TYPE="e2-standard-4"

# Disk & OS
DISK_SIZE_GB="50"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

# Firewall rule name & port
FIREWALL_RULE_NAME="allow-airflow-server"
FIREWALL_PORT="8080"
FIREWALL_SOURCE="0.0.0.0/0"

# Target tag for the firewall rule and VM
TARGET_TAG="airflow-server"

# Snapshot name (if you want to create one after)
CREATE_SNAPSHOT="no"   # change to "yes" if you want a snapshot
SNAPSHOT_NAME="arm-vm-snapshot-$(date +%Y%m%d-%H%M%S)"

# ----------- 1) FIREWALL RULE -----------
echo "🔒 Checking if firewall rule '$FIREWALL_RULE_NAME' exists..."
EXISTING_RULE=$(gcloud compute firewall-rules list \
  --filter="name=($FIREWALL_RULE_NAME)" \
  --format="value(name)")

if [[ -z "$EXISTING_RULE" ]]; then
  echo "❌ Firewall rule '$FIREWALL_RULE_NAME' not found. Creating it..."
  gcloud compute firewall-rules create "$FIREWALL_RULE_NAME" \
    --action=ALLOW \
    --direction=INGRESS \
    --rules=tcp:$FIREWALL_PORT \
    --source-ranges="$FIREWALL_SOURCE" \
    --target-tags="$TARGET_TAG" \
    --description="Allow Airflow UI on port $FIREWALL_PORT from $FIREWALL_SOURCE" \
    --priority=1000
  echo "✅ Firewall rule '$FIREWALL_RULE_NAME' created."
else
  echo "✅ Firewall rule '$FIREWALL_RULE_NAME' already exists."
fi

# ----------- 2) CREATE OR START VM -----------
echo "🔍 Checking if VM '$VM_NAME' exists..."
EXISTS=$(gcloud compute instances list \
  --filter="name=($VM_NAME) AND zone:($VM_ZONE)" \
  --format="value(name)")

if [[ -z "$EXISTS" ]]; then
  echo "❌ No VM named '$VM_NAME' found. Creating a new one..."
  gcloud compute instances create "$VM_NAME" \
    --zone "$VM_ZONE" \
    --machine-type "$MACHINE_TYPE" \
    --image-family "$IMAGE_FAMILY" \
    --image-project "$IMAGE_PROJECT" \
    --boot-disk-size "$DISK_SIZE_GB" \
    --scopes "cloud-platform" \
    --tags="$TARGET_TAG"
  echo "✅ VM '$VM_NAME' created successfully."
  echo "Waiting 60 seconds for the VM to fully spin up..."
  sleep 60

else
  echo "✅ VM '$VM_NAME' already exists. Checking status..."
  VM_STATUS=$(gcloud compute instances describe "$VM_NAME" \
    --zone "$VM_ZONE" \
    --format="value(status)")
  if [[ "$VM_STATUS" != "RUNNING" ]]; then
    echo "🔄 VM '$VM_NAME' is not running. Starting it..."
    gcloud compute instances start "$VM_NAME" --zone "$VM_ZONE"
    echo "✅ VM '$VM_NAME' started."
  else
    echo "✅ VM '$VM_NAME' is already running."
  fi
fi

# ----------- 3) (OPTIONAL) SNAPSHOT -----------
if [[ "$CREATE_SNAPSHOT" == "yes" ]]; then
  echo "🗃  Creating snapshot '$SNAPSHOT_NAME' for VM '$VM_NAME'..."
  gcloud compute disks snapshot "$VM_NAME" \
    --zone "$VM_ZONE" \
    --snapshot-names "$SNAPSHOT_NAME"
  echo "✅ Snapshot '$SNAPSHOT_NAME' created."
fi

echo "🎉 Done. Airflow Server and firewall rule are configured!"
