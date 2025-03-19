resource "google_compute_instance_template" "airflow_template" {
  name         = "airflow-template"
  machine_type = var.machine_type
  region       = var.region

  disk {
    source_image = "projects/primordial-veld-450618-n4/global/images/my-airflow-image" # Custom image
    auto_delete  = true
    boot         = true
  }

  network_interface {
    network = google_compute_network.vpc.name
    subnetwork = google_compute_subnetwork.airflow_subnet.name
    access_config {} # Public IP
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
