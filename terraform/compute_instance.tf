resource "google_compute_instance" "airflow_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
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
# Update the system and install Docker
sudo apt update -y
sudo apt install -y docker.io

# Install Docker Compose (latest version)
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/download/$(curl -s https://api.github.com/repos/docker/compose/releases/latest | jq -r .tag_name)/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Start Docker service and add user to Docker group
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Wait for Docker service to start (add a small delay)
sleep 5

# Run Docker Compose to start Airflow
docker-compose -f /opt/airflow/docker-compose.yaml up -d

EOT

  tags = ["airflow-server"]

  lifecycle {
    ignore_changes = [
      # Ignore differences in startup script if formatting or computed defaults change
      metadata_startup_script
    ]
  }
}
