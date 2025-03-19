resource "google_compute_image" "airflow_image" {
  name        = "my-airflow-image"
  source_disk = google_compute_instance.airflow_vm.self_link
  depends_on  = [google_compute_instance.airflow_vm]
}
