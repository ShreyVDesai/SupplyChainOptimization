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
    enable-oslogin = "FALSE"  # Ensure OS Login is disabled to use metadata SSH keys.
    ssh-keys       = "ubuntu:${tls_private_key.ssh_key.public_key_openssh}"
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

  lifecycle {
    ignore_changes = [
      # Ignore differences in startup script if formatting or computed defaults change
      metadata_startup_script,
      # If the computed SSH key string (or order) differs, ignore that as well
      metadata["ssh-keys"],
      # Sometimes the boot disk's source attribute is computed and might differ
      boot_disk[0].source,
    ]
  }
}
