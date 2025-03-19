output "ssh_private_key" {
  value     = tls_private_key.ssh_key.private_key_pem
  sensitive = true
}

output "vm_external_ip" {
  value = google_compute_instance.airflow_server.network_interface[0].access_config[0].nat_ip
}
