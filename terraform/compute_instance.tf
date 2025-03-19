resource "google_compute_instance" "airflow_server" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["airflow-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-2204-lts"  # Ubuntu 22.04 LTS image from ubuntu-os-cloud.
      size  = var.disk_size_gb
    }
  }

  network_interface {
    network = "default"  # Change if using a different network.
    access_config {}     # This block allocates an ephemeral public IP.
  }

  metadata = {
    ssh-keys = "ubuntu:${tls_private_key.ssh_key.public_key_openssh}"
  }
}