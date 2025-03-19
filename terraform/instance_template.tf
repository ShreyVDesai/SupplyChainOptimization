resource "google_compute_instance_template" "airflow_template" {
  name         = "airflow-template"
  machine_type = var.machine_type
  region       = var.region

  disk {
    source_image = google_compute_image.airflow_image.self_link
    auto_delete  = true
    boot         = true
  }

  network_interface {
    network    = google_compute_network.airflow_vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link
    access_config {}
  }

  metadata_startup_script = <<EOT
#!/bin/bash
sudo apt update -y
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker ubuntu
docker-compose -f /opt/airflow/docker-compose.yaml up -d
EOT
}
