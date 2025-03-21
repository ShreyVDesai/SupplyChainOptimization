# resource "tls_private_key" "ssh_key" {
#   algorithm = "RSA"
#   rsa_bits  = 4096
# }


resource "tls_private_key" "main" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

output "ssh_private_key" {
  value     = tls_private_key.main.private_key_pem
  sensitive = true
}

output "ssh_public_key" {
  value = "ubuntu:${tls_private_key.main.public_key_openssh}"
}
