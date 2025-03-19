resource "google_compute_image" "airflow_image" {
  name        = "my-airflow-image"
  source_disk = google_compute_instance.airflow_vm.boot_disk[0].device_name
  depends_on  = [google_compute_instance.airflow_vm]
}