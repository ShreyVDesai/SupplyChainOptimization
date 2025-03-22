# # #!/bin/bash
# # set -e

# # # Change into the terraform configuration directory
# # cd terraform

# # # Use environment variables provided by the workflow.
# # PROJECT_ID="${TF_VAR_project_id}"
# # REGION="${TF_VAR_region}"
# # ARTIFACT_REPO="${ARTIFACT_REGISTRY_NAME}"
# # VM_NAME="${VM_NAME}"        # Provided by your workflow environment
# # VM_ZONE="${VM_ZONE}"        # Provided by your workflow environment
# # FIREWALL_NAME="allow-airflow-server"

# # echo "Checking if Artifact Registry exists..."
# # artifact_exists=$(gcloud artifacts repositories list \
# #   --project="${PROJECT_ID}" \
# #   --location="${REGION}" \
# #   --filter="name:projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}" \
# #   --format="value(name)")

# # if [ -n "$artifact_exists" ]; then
# #   echo "Artifact Registry exists. Importing it into Terraform..."
# #   terraform import google_artifact_registry_repository.airflow_docker "projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}"
# # else
# #   echo "Artifact Registry does not exist. Terraform will create it."
# # fi

# # echo "Checking if Firewall exists..."
# # firewall_exists=$(gcloud compute firewall-rules list \
# #   --project="${PROJECT_ID}" \
# #   --filter="name=${FIREWALL_NAME}" \
# #   --format="value(name)")

# # if [ -n "$firewall_exists" ]; then
# #   echo "Firewall exists. Importing it into Terraform..."
# #   terraform import google_compute_firewall.airflow_server "projects/${PROJECT_ID}/global/firewalls/${FIREWALL_NAME}"
# # else
# #   echo "Firewall does not exist. Terraform will create it."
# # fi

# # echo "Checking if Compute Instance exists..."
# # instance_exists=$(gcloud compute instances list \
# #   --project="${PROJECT_ID}" \
# #   --zones="${VM_ZONE}" \
# #   --filter="name=${VM_NAME}" \
# #   --format="value(name)")

# # if [ -n "$instance_exists" ]; then
# #   echo "Compute Instance exists. Importing it into Terraform..."
# #   terraform import google_compute_instance.airflow_server "projects/${PROJECT_ID}/zones/${VM_ZONE}/instances/${VM_NAME}"
# # else
# #   echo "Compute Instance does not exist. Terraform will create it."
# # fi

# # echo "Running Terraform plan and apply..."
# # terraform plan -out=tfplan
# # terraform apply -auto-approve tfplan

# # echo "Waiting for VM to fully initialize..."
# # sleep 60
# #!/bin/bash



# #!/bin/bash
# set -e

# # Change into the Terraform configuration directory
# cd terraform

# # Use environment variables provided by the workflow.
# PROJECT_ID="${TF_VAR_project_id}"
# REGION="${TF_VAR_region}"
# ARTIFACT_REPO="${ARTIFACT_REGISTRY_NAME}"
# VM_NAME="${VM_NAME}"
# VM_ZONE="${VM_ZONE}"
# FIREWALL_NAME="allow-airflow-server"
# VPC_NAME="airflow-vpc"
# SUBNET_NAME="airflow-subnet"
# LOAD_BALANCER_NAME="airflow-load-balancer"
# HEALTH_CHECK_NAME="airflow-health-check"
# BACKEND_SERVICE_NAME="airflow-backend"
# IMAGE_NAME="my-airflow-image"

# # Note: We are not updating gcloud components because the component manager is disabled.

# # Function to import resources into Terraform if they exist.
# import_if_exists() {
#   local resource_type=$1
#   local resource_name=$2
#   local gcloud_command=$3
#   local terraform_import_command=$4

#   echo "Checking if $resource_type: $resource_name exists..."
#   resource_exists=$(eval "$gcloud_command")

#   if [ -n "$resource_exists" ]; then
#     echo "$resource_type: $resource_name exists. Importing into Terraform..."
#     eval "$terraform_import_command"
#   else
#     echo "$resource_type: $resource_name does not exist. Terraform will create it."
#   fi
# }

# ### Artifact Registry
# import_if_exists "Artifact Registry" "$ARTIFACT_REPO" \
#   "gcloud artifacts repositories list --project=${PROJECT_ID} --location=${REGION} --format='value(name)' | grep -w ${ARTIFACT_REPO}" \
#   "terraform import google_artifact_registry_repository.airflow_docker projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}"

# ### Firewall Rule: allow-airflow-server
# import_if_exists "Firewall Rule" "$FIREWALL_NAME" \
#   "gcloud compute firewall-rules list --project=${PROJECT_ID} --filter='name=${FIREWALL_NAME}' --format='value(name)'" \
#   "terraform import google_compute_firewall.airflow_server projects/${PROJECT_ID}/global/firewalls/${FIREWALL_NAME}"

# ### Firewall Rule: allow-http
# import_if_exists "Firewall Rule" "allow-http" \
#   "gcloud compute firewall-rules list --project=${PROJECT_ID} --filter='name=allow-http' --format='value(name)'" \
#   "terraform import google_compute_firewall.allow_http projects/${PROJECT_ID}/global/firewalls/allow-http"

# ### Compute Instance (VM)
# import_if_exists "Compute Instance" "$VM_NAME" \
#   "gcloud compute instances list --project=${PROJECT_ID} --zones=${VM_ZONE} --filter='name=${VM_NAME}' --format='value(name)'" \
#   "terraform import google_compute_instance.airflow_vm projects/${PROJECT_ID}/zones/${VM_ZONE}/instances/${VM_NAME}"

# ### VPC
# import_if_exists "VPC" "$VPC_NAME" \
#   "gcloud compute networks list --project=${PROJECT_ID} --filter='name=${VPC_NAME}' --format='value(name)'" \
#   "terraform import google_compute_network.airflow_vpc projects/${PROJECT_ID}/global/networks/${VPC_NAME}"

# ### Subnet
# import_if_exists "Subnet" "$SUBNET_NAME" \
#   "gcloud compute networks subnets list --project=${PROJECT_ID} --filter='name=${SUBNET_NAME}' --format='value(name)'" \
#   "terraform import google_compute_subnetwork.airflow_subnet projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${SUBNET_NAME}"

# ### Load Balancer (Forwarding Rule)
# import_if_exists "Load Balancer" "$LOAD_BALANCER_NAME" \
#   "gcloud compute forwarding-rules list --project=${PROJECT_ID} --filter='name=${LOAD_BALANCER_NAME}' --format='value(name)'" \
#   "terraform import google_compute_global_forwarding_rule.airflow_forwarding_rule projects/${PROJECT_ID}/global/forwardingRules/${LOAD_BALANCER_NAME}"

# ### Backend Service for Load Balancer
# import_if_exists "Backend Service" "$BACKEND_SERVICE_NAME" \
#   "gcloud compute backend-services list --project=${PROJECT_ID} --filter='name=${BACKEND_SERVICE_NAME}' --format='value(name)'" \
#   "terraform import google_compute_backend_service.airflow_backend projects/${PROJECT_ID}/global/backendServices/${BACKEND_SERVICE_NAME}"

# ### Health Check for Load Balancer
# import_if_exists "Health Check" "$HEALTH_CHECK_NAME" \
#   "gcloud compute health-checks list --project=${PROJECT_ID} --filter='name=${HEALTH_CHECK_NAME}' --format='value(name)'" \
#   "terraform import google_compute_health_check.airflow_health_check projects/${PROJECT_ID}/global/healthChecks/${HEALTH_CHECK_NAME}"


# echo "Checking if Autoscaler: airflow-autoscaler exists..."
# autoscaler_exists=$(gcloud compute instance-groups managed describe airflow-mig \
#   --zone="${VM_ZONE}" \
#   --project="${PROJECT_ID}" \
#   --format="value(autoscaler)" 2>/dev/null || echo "")

# if [[ -n "$autoscaler_exists" ]]; then
#   echo "Autoscaler airflow-autoscaler exists. Importing into Terraform..."
#   terraform import google_compute_autoscaler.airflow_autoscaler \
#     projects/${PROJECT_ID}/zones/${VM_ZONE}/autoscalers/airflow-autoscaler || echo "⚠️ Terraform import failed."
# else
#   echo "Autoscaler airflow-autoscaler does not exist in ${VM_ZONE}. Terraform will create it."
# fi



# ### URL Map
# import_if_exists "URL Map" "airflow-url-map" \
#   "gcloud compute url-maps list --project=${PROJECT_ID} --filter='name=airflow-url-map' --format='value(name)'" \
#   "terraform import google_compute_url_map.airflow_url_map projects/${PROJECT_ID}/global/urlMaps/airflow-url-map"


# ### Target HTTP Proxy
# import_if_exists "Target HTTP Proxy" "airflow-http-proxy" \
#   "gcloud compute target-http-proxies list --project=${PROJECT_ID} --filter='name=airflow-http-proxy' --format='value(name)'" \
#   "terraform import google_compute_target_http_proxy.airflow_http_proxy projects/${PROJECT_ID}/global/targetHttpProxies/airflow-http-proxy"


# ### Global Forwarding Rule
# import_if_exists "Global Forwarding Rule" "airflow-forwarding-rule" \
#   "gcloud compute forwarding-rules list --global --project=${PROJECT_ID} --filter='name=airflow-forwarding-rule' --format='value(name)'" \
#   "terraform import google_compute_global_forwarding_rule.airflow_forwarding_rule projects/${PROJECT_ID}/global/forwardingRules/airflow-forwarding-rule"


# ### Instance Group Manager
# import_if_exists "Instance Group" "airflow-mig" \
#   "gcloud compute instance-groups managed list --project=${PROJECT_ID} --zones=${VM_ZONE} --filter='name=airflow-mig' --format='value(name)' 2>/dev/null" \
#   "terraform import google_compute_instance_group_manager.airflow_mig projects/${PROJECT_ID}/zones/${VM_ZONE}/instanceGroupManagers/airflow-mig"

# ### Image
# import_if_exists "Image" "$IMAGE_NAME" \
#   "gcloud compute images list --project=${PROJECT_ID} --filter='name=${IMAGE_NAME}' --format='value(name)'" \
#   "terraform import google_compute_image.airflow_image projects/${PROJECT_ID}/global/images/${IMAGE_NAME}"

# ### Instance Template
# import_if_exists "Instance Template" "airflow-template" \
#   "gcloud compute instance-templates list --project=${PROJECT_ID} --filter='name=airflow-template' --format='value(name)'" \
#   "terraform import google_compute_instance_template.airflow_template projects/${PROJECT_ID}/global/instanceTemplates/airflow-template"

# echo "Running Terraform plan and apply..."
# terraform plan -out=tfplan
# terraform apply -auto-approve tfplan

# # Wait for VM to be fully initialized
# echo "Waiting for VM to fully initialize..."
# sleep 60








#!/bin/bash
set -e

# Change into the Terraform configuration directory
cd terraform

# Use environment variables provided by the workflow.
PROJECT_ID="${TF_VAR_project_id}"
REGION="${TF_VAR_region}"
ARTIFACT_REPO="${ARTIFACT_REGISTRY_NAME}"
# Production instance name remains as VM_NAME.
VM_NAME="${VM_NAME}"
# Baker instance name (new) – defaulting to "airflow-server-baker"
BAKER_VM_NAME="${BAKER_VM_NAME:-airflow-server-baker}"
VM_ZONE="${VM_ZONE}"
FIREWALL_NAME="allow-airflow-server"
VPC_NAME="airflow-vpc"
SUBNET_NAME="airflow-subnet"
LOAD_BALANCER_NAME="airflow-load-balancer"
HEALTH_CHECK_NAME="airflow-health-check"
BACKEND_SERVICE_NAME="airflow-backend"
# Use IMAGE_NAME for the custom image (ensure it matches your Terraform config)
IMAGE_NAME="my-airflow-image"

# Note: We are not updating gcloud components because the component manager is disabled.

# Function to import resources into Terraform if they exist.
import_if_exists() {
  local resource_type=$1
  local resource_name=$2
  local gcloud_command=$3
  local terraform_import_command=$4

  echo "Checking if $resource_type: $resource_name exists..."
  resource_exists=$(eval "$gcloud_command")

  if [ -n "$resource_exists" ]; then
    echo "$resource_type: $resource_name exists. Importing into Terraform..."
    eval "$terraform_import_command"
  else
    echo "$resource_type: $resource_name does not exist. Terraform will create it."
  fi
}

### Artifact Registry
import_if_exists "Artifact Registry" "$ARTIFACT_REPO" \
  "gcloud artifacts repositories list --project=${PROJECT_ID} --location=${REGION} --format='value(name)' | grep -w ${ARTIFACT_REPO}" \
  "terraform import google_artifact_registry_repository.airflow_docker projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}"

### Firewall Rule: allow-airflow-server
import_if_exists "Firewall Rule" "$FIREWALL_NAME" \
  "gcloud compute firewall-rules list --project=${PROJECT_ID} --filter='name=${FIREWALL_NAME}' --format='value(name)'" \
  "terraform import google_compute_firewall.airflow_server projects/${PROJECT_ID}/global/firewalls/${FIREWALL_NAME}"

### Firewall Rule: allow-http
import_if_exists "Firewall Rule" "allow-http" \
  "gcloud compute firewall-rules list --project=${PROJECT_ID} --filter='name=allow-http' --format='value(name)'" \
  "terraform import google_compute_firewall.allow_http projects/${PROJECT_ID}/global/firewalls/allow-http"

### Compute Instance (Production)
import_if_exists "Compute Instance" "$VM_NAME" \
  "gcloud compute instances list --project=${PROJECT_ID} --zones=${VM_ZONE} --filter='name=${VM_NAME}' --format='value(name)'" \
  "terraform import google_compute_instance.airflow_vm projects/${PROJECT_ID}/zones/${VM_ZONE}/instances/${VM_NAME}"

### Compute Instance (Baker)
import_if_exists "Compute Baker Instance" "$BAKER_VM_NAME" \
  "gcloud compute instances list --project=${PROJECT_ID} --zones=${VM_ZONE} --filter='name=${BAKER_VM_NAME}' --format='value(name)'" \
  "terraform import google_compute_instance.airflow_baker projects/${PROJECT_ID}/zones/${VM_ZONE}/instances/${BAKER_VM_NAME}"

### VPC
import_if_exists "VPC" "$VPC_NAME" \
  "gcloud compute networks list --project=${PROJECT_ID} --filter='name=${VPC_NAME}' --format='value(name)'" \
  "terraform import google_compute_network.airflow_vpc projects/${PROJECT_ID}/global/networks/${VPC_NAME}"

### Subnet
import_if_exists "Subnet" "$SUBNET_NAME" \
  "gcloud compute networks subnets list --project=${PROJECT_ID} --filter='name=${SUBNET_NAME}' --format='value(name)'" \
  "terraform import google_compute_subnetwork.airflow_subnet projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${SUBNET_NAME}"

### Load Balancer (Forwarding Rule)
import_if_exists "Load Balancer" "$LOAD_BALANCER_NAME" \
  "gcloud compute forwarding-rules list --project=${PROJECT_ID} --filter='name=${LOAD_BALANCER_NAME}' --format='value(name)'" \
  "terraform import google_compute_global_forwarding_rule.airflow_forwarding_rule projects/${PROJECT_ID}/global/forwardingRules/${LOAD_BALANCER_NAME}"

### Backend Service for Load Balancer
import_if_exists "Backend Service" "$BACKEND_SERVICE_NAME" \
  "gcloud compute backend-services list --project=${PROJECT_ID} --filter='name=${BACKEND_SERVICE_NAME}' --format='value(name)'" \
  "terraform import google_compute_backend_service.airflow_backend projects/${PROJECT_ID}/global/backendServices/${BACKEND_SERVICE_NAME}"

### Health Check for Load Balancer
import_if_exists "Health Check" "$HEALTH_CHECK_NAME" \
  "gcloud compute health-checks list --project=${PROJECT_ID} --filter='name=${HEALTH_CHECK_NAME}' --format='value(name)'" \
  "terraform import google_compute_health_check.airflow_health_check projects/${PROJECT_ID}/global/healthChecks/${HEALTH_CHECK_NAME}"

echo "Checking if Autoscaler: airflow-autoscaler exists..."
autoscaler_exists=$(gcloud compute instance-groups managed describe airflow-mig \
  --zone="${VM_ZONE}" \
  --project="${PROJECT_ID}" \
  --format="value(autoscaler)" 2>/dev/null || echo "")

if [[ -n "$autoscaler_exists" ]]; then
  echo "Autoscaler airflow-autoscaler exists. Importing into Terraform..."
  terraform import google_compute_autoscaler.airflow_autoscaler \
    projects/${PROJECT_ID}/zones/${VM_ZONE}/autoscalers/airflow-autoscaler || echo "⚠️ Terraform import failed."
else
  echo "Autoscaler airflow-autoscaler does not exist in ${VM_ZONE}. Terraform will create it."
fi

### URL Map
import_if_exists "URL Map" "airflow-url-map" \
  "gcloud compute url-maps list --project=${PROJECT_ID} --filter='name=airflow-url-map' --format='value(name)'" \
  "terraform import google_compute_url_map.airflow_url_map projects/${PROJECT_ID}/global/urlMaps/airflow-url-map"

### Target HTTP Proxy
import_if_exists "Target HTTP Proxy" "airflow-http-proxy" \
  "gcloud compute target-http-proxies list --project=${PROJECT_ID} --filter='name=airflow-http-proxy' --format='value(name)'" \
  "terraform import google_compute_target_http_proxy.airflow_http_proxy projects/${PROJECT_ID}/global/targetHttpProxies/airflow-http-proxy"

### Global Forwarding Rule
import_if_exists "Global Forwarding Rule" "airflow-forwarding-rule" \
  "gcloud compute forwarding-rules list --global --project=${PROJECT_ID} --filter='name=airflow-forwarding-rule' --format='value(name)'" \
  "terraform import google_compute_global_forwarding_rule.airflow_forwarding_rule projects/${PROJECT_ID}/global/forwardingRules/airflow-forwarding-rule"

### Instance Group Manager
import_if_exists "Instance Group" "airflow-mig" \
  "gcloud compute instance-groups managed list --project=${PROJECT_ID} --zones=${VM_ZONE} --filter='name=airflow-mig' --format='value(name)' 2>/dev/null" \
  "terraform import google_compute_instance_group_manager.airflow_mig projects/${PROJECT_ID}/zones/${VM_ZONE}/instanceGroupManagers/airflow-mig"

### Custom Image (Baked from Baker Instance)
import_if_exists "Custom Image" "$IMAGE_NAME" \
  "gcloud compute images list --project=${PROJECT_ID} --filter='name=${IMAGE_NAME}' --format='value(name)'" \
  "terraform import google_compute_image.airflow_custom_image projects/${PROJECT_ID}/global/images/${IMAGE_NAME}"

### Instance Template
import_if_exists "Instance Template" "airflow-template" \
  "gcloud compute instance-templates list --project=${PROJECT_ID} --filter='name=airflow-template' --format='value(name)'" \
  "terraform import google_compute_instance_template.airflow_template projects/${PROJECT_ID}/global/instanceTemplates/airflow-template"

echo "Running Terraform plan and apply..."
terraform plan -out=tfplan
terraform apply -auto-approve tfplan

# Wait for VM to be fully initialized
echo "Waiting for VM to fully initialize..."
sleep 60
