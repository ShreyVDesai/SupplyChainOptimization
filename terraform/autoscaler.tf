resource "google_compute_autoscaler" "airflow_autoscaler" {
  name = "airflow-autoscaler"
  zone = var.zone

  target = google_compute_instance_group_manager.airflow_mig.self_link

  autoscaling_policy {
    min_replicas = 1
    max_replicas = 5
    cooldown_period = 60

    cpu_utilization {
      target = 0.6  # 60% CPU utilization threshold
    }
  }

  lifecycle {
    ignore_changes = [autoscaling_policy]  # Use with caution—ensure you really want to ignore updates here.
  }
}