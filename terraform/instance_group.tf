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

resource "google_compute_autoscaler" "airflow_autoscaler" {
  name   = "airflow-autoscaler"
  target = google_compute_instance_group_manager.airflow_mig.id

  autoscaling_policy {
    max_replicas    = 5
    min_replicas    = 1
    cooldown_period = 60

    cpu_utilization {
      target = 0.6
    }
  }
}
