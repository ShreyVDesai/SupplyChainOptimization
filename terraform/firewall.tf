resource "google_compute_firewall" "airflow_server" {
  name    = "allow-airflow-server"
  network = "default"  # Change if you use a custom VPC.
  
  allow {
    protocol = "tcp"
    ports    = ["22", "8080"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["airflow-server"]
}