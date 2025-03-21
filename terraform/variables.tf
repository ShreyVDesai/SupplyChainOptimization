variable "project_id" {
  description = "GCP project id"
  type        = string
  default = "primordial-veld-450618-n4"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "artifact_registry_name" {
  description = "Name of the Artifact Registry repository"
  type        = string
  default     = "airflow-docker-image"
}

variable "vm_name" {
  description = "Name of the Airflow server VM"
  type        = string
  default     = "airflow-server"
}

variable "machine_type" {
  description = "GCP machine type"
  type        = string
  default     = "e2-standard-4"
}

variable "disk_size_gb" {
  description = "Boot disk size for the VM in GB"
  type        = number
  default     = 50
}