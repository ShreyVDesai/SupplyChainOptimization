resource "google_compute_instance" "airflow_baker" {
  name         = "${var.vm_name}-baker"
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
    access_config {}
  }

  metadata = {
    block-project-ssh-keys = "true"
    ssh-keys               = "ubuntu:${tls_private_key.main.public_key_openssh}"
  }

  # Optionally add a startup script or run your sync_files.sh externally
  metadata_startup_script = <<EOT
#!/bin/bash
# (Optional) You can trigger file sync here if needed.
EOT

  tags = ["airflow-baker"]

  lifecycle {
    ignore_changes = [
      metadata_startup_script
    ]
  }
}