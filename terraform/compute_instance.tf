resource "google_compute_instance" "airflow_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts" # Base image to create VM
      size  = 50
    }
  }

  network_interface {
    network    = google_compute_network.vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link
    access_config {} # Assigns a public IP
  }

  metadata_startup_script = <<EOT
#!/bin/bash
sudo apt update -y
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker ubuntu
docker-compose -f /opt/airflow/docker-compose.yaml up -d
EOT

  tags = ["airflow-server"]
}

# **Create Image only if it doesn't exist**
resource "google_compute_image" "airflow_image" {
  name       = "my-airflow-image"
  source_disk = google_compute_instance.airflow_vm.boot_disk[0].device_name
  depends_on = [google_compute_instance.airflow_vm]
}
