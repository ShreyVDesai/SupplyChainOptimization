resource "google_artifact_registry_repository" "airflow_docker" {
  provider      = google-beta
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_name
  format        = "DOCKER"
  description   = "Docker repository for data-pipeline"
}