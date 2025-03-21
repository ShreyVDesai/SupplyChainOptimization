resource "google_compute_instance_group_manager" "airflow_mig" {
  name               = "airflow-mig"
  zone               = var.zone          # e.g., "us-central1-a"
  base_instance_name = "airflow-instance"
  target_size        = 1

  version {
    instance_template = google_compute_instance_template.airflow_template.self_link
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.airflow_health_check.self_link
    initial_delay_sec = 300
  }
}
