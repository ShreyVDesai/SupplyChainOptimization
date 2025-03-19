resource "google_compute_image" "airflow_image" {
  name        = "my-airflow-image"
  source_disk = "projects/${var.project_id}/zones/${var.zone}/disks/${google_compute_instance.airflow_vm.boot_disk[0].device_name}"
  depends_on  = [google_compute_instance.airflow_vm]
}
