output "ssh_private_key" {
  value     = tls_private_key.ssh_key.private_key_pem
  sensitive = true
}

output "ssh_public_key" {
  value = tls_private_key.ssh_key.public_key_openssh
}

output "vm_external_ip" {
  value = google_compute_instance.airflow_vm.network_interface[0].access_config[0].nat_ip
}
