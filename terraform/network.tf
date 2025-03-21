resource "google_compute_network" "airflow_vpc" {
  name                    = "airflow-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "airflow_subnet" {
  name          = "airflow-subnet"
  network       = google_compute_network.airflow_vpc.self_link
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
}

resource "google_compute_firewall" "allow_http" {
  name    = "allow-http"
  network = google_compute_network.airflow_vpc.self_link

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
}
