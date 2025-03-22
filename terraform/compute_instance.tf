resource "google_compute_instance" "airflow_vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
    }
  }

  network_interface {
    network    = google_compute_network.airflow_vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link
    access_config {} # Assigns a public IP
  }

  metadata = {
    block-project-ssh-keys = "true"
    ssh-keys               = "ubuntu:${tls_private_key.main.public_key_openssh}"
  }

  metadata_startup_script = <<EOT
#!/bin/bash
# Update the system and install Docker if it is not present
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing..."
    sudo apt-get update -y
    echo "🚀 Adding Docker repository..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable"
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
else
    echo "✅ Docker is already installed."
fi

# Install Docker Compose v2 as a Docker CLI plugin so that "docker compose" works
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose plugin not found. Installing latest version..."
    DOCKER_COMPOSE_VERSION=\$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL "https://github.com/docker/compose/releases/download/v\${DOCKER_COMPOSE_VERSION}/docker-compose-linux-\$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
else
    echo "✅ Docker Compose plugin is already installed."
fi

# Give user Docker permissions
echo "🔄 Adding user to Docker group..."
sudo usermod -aG docker \$USER
sudo systemctl restart docker
echo "✅ User added to Docker group and Docker restarted."

# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock
echo "✅ Docker socket permissions fixed."
EOT

  tags = ["airflow-server"]

  lifecycle {
    ignore_changes = [
      # Ignore differences in startup script if formatting or computed defaults change
      metadata_startup_script
    ]
  }
}
