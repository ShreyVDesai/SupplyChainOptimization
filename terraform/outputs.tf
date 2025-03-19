output "vm_external_ip" {
  description = "The external IP address of the Airflow server VM."
  value       = google_compute_instance.airflow_server.network_interface[0].access_config[0].nat_ip
}

output "ssh_private_key" {
  value     = tls_private_key.ssh_key.private_key_pem
  sensitive = true
}
