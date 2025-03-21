resource "google_compute_backend_service" "airflow_backend" {
  name                           = "airflow-backend"
  protocol                       = "HTTP"
  timeout_sec                    = 30
  load_balancing_scheme          = "EXTERNAL"
  connection_draining_timeout_sec = 300
  health_checks                  = [
    google_compute_health_check.airflow_health_check.self_link,
  ]

  backend {
    group = google_compute_instance_group_manager.airflow_mig.instance_group
    balancing_mode = "UTILIZATION"
    capacity_scaler = 1
  }
}

resource "google_compute_health_check" "airflow_health_check" {
  name               = "airflow-health-check"
  check_interval_sec = 10
  timeout_sec        = 5

  http_health_check {
    port = 8080
  }
}

resource "google_compute_url_map" "airflow_url_map" {
  name            = "airflow-url-map"
  default_service = google_compute_backend_service.airflow_backend.self_link
}

resource "google_compute_target_http_proxy" "airflow_http_proxy" {
  name    = "airflow-http-proxy"
  url_map = google_compute_url_map.airflow_url_map.self_link
}

resource "google_compute_global_forwarding_rule" "airflow_forwarding_rule" {
  name       = "airflow-forwarding-rule"
  target     = google_compute_target_http_proxy.airflow_http_proxy.self_link
  port_range = "80"
}