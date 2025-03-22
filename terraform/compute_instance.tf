resource "google_compute_instance" "airflow_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = google_compute_image.airflow_image.self_link
      size  = 50
    }
  }

  network_interface {
    network    = google_compute_network.airflow_vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link
    access_config {} # Assigns a public IP
  }

  metadata = {
    block-project-ssh-keys = "true"
    ssh-keys               = "ubuntu:${tls_private_key.main.public_key_openssh}"
  }

  metadata_startup_script = <<EOT
#!/bin/bash
# Change to the directory where your docker-compose.yml file is located.
cd /opt/airflow

gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker compose pull || true

echo "🚀 Stopping any running containers..."
docker compose down || true

# Remove postgres volume if you want to reset the DB (warning: this clears data)
docker volume rm airflow_postgres-db-volume || true

echo "🚀 Starting Airflow using Docker Compose..."
docker compose up -d --remove-orphans
EOT

  tags = ["airflow-server"]

  lifecycle {
    ignore_changes = [
      metadata_startup_script
    ]
  }
}
