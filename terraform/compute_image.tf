# Stop the VM using a null_resource.
resource "null_resource" "stop_airflow_vm" {
  depends_on = [google_compute_instance.airflow_vm]

  provisioner "local-exec" {
    command = "gcloud compute instances stop ${google_compute_instance.airflow_vm.name} --zone ${var.zone}"
  }
}

# Optional: Wait for a while to ensure the instance has fully stopped.
resource "time_sleep" "wait_for_instance_stop" {
  depends_on     = [null_resource.stop_airflow_vm]
  create_duration = "60s"
}

# Create the image after the VM is stopped.
resource "google_compute_image" "airflow_image" {
  name        = "my-airflow-image"
  source_disk = google_compute_instance.airflow_vm.boot_disk[0].source
  depends_on  = [time_sleep.wait_for_instance_stop]

  lifecycle {
    ignore_changes = [source_disk]
  }
}

# Start the VM again after the image is created.
resource "null_resource" "start_airflow_vm" {
  depends_on = [google_compute_image.airflow_image]

  provisioner "local-exec" {
    command = "gcloud compute instances start ${google_compute_instance.airflow_vm.name} --zone ${var.zone}"
  }
}