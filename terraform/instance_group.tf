resource "google_compute_instance_group_manager" "airflow_mig" {
  name               = "airflow-mig"
  base_instance_name = "airflow-instance"

  version {
    instance_template = google_compute_instance_template.airflow_template.self_link
  }

  target_size = 1

  auto_healing_policies {
    health_check      = google_compute_health_check.airflow_health_check.id
    initial_delay_sec = 300
  }
}
