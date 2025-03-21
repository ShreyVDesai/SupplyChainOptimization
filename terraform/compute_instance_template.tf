resource "google_compute_instance_template" "airflow_template" {
  name         = "airflow-template"
  machine_type = var.machine_type

  disk {
    auto_delete  = true
    boot         = true
    source_image = google_compute_image.airflow_image.self_link
  }

  network_interface {
    network    = google_compute_network.airflow_vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link
    access_config {}
  }

  # Add any other required configuration here.

  lifecycle {
    create_before_destroy = true
    ignore_changes = [
      metadata_startup_script,
    ]
  }
}
