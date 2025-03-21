# resource "tls_private_key" "ssh_key" {
#   algorithm = "RSA"
#   rsa_bits  = 4096
# }


resource "tls_private_key" "main" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

