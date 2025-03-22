# Stop the baker instance to capture a consistent disk state.
resource "null_resource" "stop_airflow_baker" {
  depends_on = [google_compute_instance.airflow_baker]
  provisioner "local-exec" {
    command = "gcloud compute instances stop ${google_compute_instance.airflow_baker.name} --zone ${var.zone}"
  }
}

resource "time_sleep" "wait_for_baker_stop" {
  depends_on      = [null_resource.stop_airflow_baker]
  create_duration = "60s"
}

resource "google_compute_image" "airflow_image" {
  name = "airflow-custom-image-${replace(replace(replace(timestamp(), ":", ""), "T", "-"), "Z", "")}"
  source_disk = google_compute_instance.airflow_baker.boot_disk[0].source
  family      = "airflow-family"
  description = "Custom image with synced Airflow files and configuration"
  depends_on  = [time_sleep.wait_for_baker_stop]
}

# Start the baker instance again after the image is captured.
resource "null_resource" "start_airflow_baker" {
  depends_on = [google_compute_image.airflow_image]
  provisioner "local-exec" {
    command = "gcloud compute instances start ${google_compute_instance.airflow_baker.name} --zone ${var.zone}"
  }
}
