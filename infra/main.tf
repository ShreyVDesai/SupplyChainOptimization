module "artifact_registry" {
  source = "./artifact_registry"
}

module "compute_instance" {
  source = "./compute_instance"
}

module "firewall" {
  source = "./firewall"
}