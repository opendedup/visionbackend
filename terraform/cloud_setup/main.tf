variable "location" {
  type = "string"
  description = "The region for the GCP project (Defaults to us-west1)"
  default = "us-west1"
}

variable "gcp_organization_id" {
  type = "string"
  default = ""
  description = "Leave this blank if you don't know what it is. The Organization id (number) under which to put the project"
}

variable "gcp_billing_account" {
  type = "string"
  default = "My Billing Account"
  description = "The billing account for the GCP Project. If left blank will default to \"My Billing Account\""
}


data "google_billing_account" "acct" {
  display_name = "${var.gcp_billing_account}"
  open         = true
}



variable "project" {
  description="The name for the GCP project where everything will run"
  type = "string"
  default ="svision"
}
resource "google_project" "flexiblevision" {
  name = "flexible Vision Project"
  project_id = "${var.project}-${random_id.project_id.hex}"
  billing_account = "${data.google_billing_account.acct.id}"
  org_id = "${var.gcp_organization_id}"
}

resource "google_service_account" "svision_service_acct" {
  project = "${google_project.flexiblevision.project_id}"
  account_id   = "svision"
  display_name = "Flexible Node Vision Editor"
}


resource "google_project_iam_member" "svision_permissions" {
  project = "${google_project.flexiblevision.number}"
  role    = "roles/editor"
  member  = "serviceAccount:${google_service_account.svision_service_acct.email}"
}

resource "google_service_account_key" "svision_key" {
  service_account_id = "${google_service_account.svision_service_acct.name}"
}

resource "google_project_service" "storage" {
  project = "${google_project.flexiblevision.number}"
  service   = "storage-api.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudresourcemanager" {
  project = "${google_project.flexiblevision.number}"
  service   = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iam" {
  project = "${google_project.flexiblevision.number}"
  service   = "iam.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "container" {
  project = "${google_project.flexiblevision.number}"
  service   = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "ml" {
  project = "${google_project.flexiblevision.number}"
  service   = "ml.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "serviceusage" {
  project = "${google_project.flexiblevision.number}"
  service   = "serviceusage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  project = "${google_project.flexiblevision.number}"
  service   = "compute.googleapis.com",
  disable_on_destroy = false
}

resource "google_project_service" "oslogin" {
  project = "${google_project.flexiblevision.number}"
  service   = "oslogin.googleapis.com"
  disable_on_destroy = false
}

resource "random_id" "instance_id" {
 byte_length = 8
}

resource "random_id" "bucket_id" {
 byte_length = 16
}

resource "random_id" "project_id" {
 byte_length = 8
}

resource "google_compute_disk" "data_disk" {
  project = "${google_project.flexiblevision.project_id}"
  name  = "sv-data-disk"
  type  = "pd-ssd"
  size = 300
  zone  = "${var.location}-a"
  labels = {
    environment = "flexiblevision"
  }
  depends_on = ["google_service_account.svision_service_acct","google_project_iam_member.svision_permissions","google_project_service.compute"]
  physical_block_size_bytes = 4096
}

resource "google_compute_instance" "default" {
  project = "${google_project.flexiblevision.project_id}"
  name         = "svserver"
  machine_type = "n1-standard-16"
  zone         = "${var.location}-a"

  tags = ["prepapi", "svision","rabbitmq"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-1804-lts"
      size = 100
      type  = "pd-ssd"
    }
  }

  

  network_interface {
    network = "default"

    access_config {
      // Ephemeral IP
    }
  }

  attached_disk {
        source      = "${google_compute_disk.data_disk.self_link}"
        device_name = "data"
        mode = "READ_WRITE"
  }

  metadata = {
    sshKeys = "ubuntu:${file("${path.module}/id_rsa.pub")}"
  }

  depends_on = ["google_service_account.svision_service_acct","google_project_iam_member.svision_permissions"]

  provisioner "file" {
    content = "${base64decode(google_service_account_key.svision_key.private_key)}"
    destination = "/home/ubuntu/svision_creds.json"
    connection {
      type= "ssh"
      user="ubuntu"
      private_key="${file("${path.module}/id_rsa")}"
    }

  }



  provisioner "file" {
    source      = "${path.module}/resources/cloud_setup.sh"
    destination = "/tmp/cloud_setup.sh"
    connection {
      type= "ssh"
      user="ubuntu"
      private_key="${file("${path.module}/id_rsa")}"
    }
  }

  

  provisioner "remote-exec" {
    inline = [
      "chmod +x /tmp/cloud_setup.sh",
      "/tmp/cloud_setup.sh ${var.project} ${google_storage_bucket.project-bucket.name} ${random_string.password.result}"
      ]
    connection {
      type= "ssh"
      user="ubuntu"
      private_key="${file("${path.module}/id_rsa")}"
    }
    
  }
  

  

  

  service_account {
    scopes = ["userinfo-email", "compute-rw", "cloud-platform","monitoring-write","logging-write","pubsub","service-control","service-management","https://www.googleapis.com/auth/trace.append"]
  }
}


resource "google_storage_bucket" "project-bucket" {
  project = "${google_project.flexiblevision.number}"
  name     = "svision-${random_id.bucket_id.hex}"
  storage_class = "REGIONAL"
  force_destroy = true
  location = "${var.location}"
}

resource "google_storage_bucket_object" "upload_models" {
  name   = "trained_models/model.config"
  source = "${path.module}/resources/model.config"
  bucket = "${google_storage_bucket.project-bucket.name}"
}

resource "google_storage_bucket_object" "upload_projects" {
  name   = "projects/projects.json"
  source = "${path.module}/resources/projects.json"
  content_type = "application/json"
  bucket = "${google_storage_bucket.project-bucket.name}"
}

resource "google_compute_firewall" "svision_firewall" {
  project = "${google_project.flexiblevision.project_id}"
  name = "svision-firewall"
  network = "default"
  depends_on = ["google_service_account.svision_service_acct","google_project_iam_member.svision_permissions","google_project_service.compute","google_compute_instance.default"]
  allow {
    protocol = "tcp"
    ports = ["80", "5672"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags = ["svision"]
}

resource "local_file" "creds" {
    content     = "${base64decode(google_service_account_key.svision_key.private_key)}"
    filename = "${path.module}/config/svision_creds.json"
}

resource "random_string" "password" {
  length = 16
  special= true
  override_special= "!@#$%_-"
}



output "instance_ip" {
 value = "\"${google_compute_instance.default.network_interface.0.access_config.0.nat_ip}\""
}

output "bucket_name" {
 value = "\"${google_storage_bucket.project-bucket.name}\""
}

output "project_id" {
 value = "\"${google_project.flexiblevision.project_id}\""
}

output "cloud_password" {
 value = "\"${random_string.password.result}\""
}

output "project_number" {
  value = "\"${google_project.flexiblevision.number}\""
}

output "service_account_name" {
  value = "\"${google_service_account.svision_service_acct.name}\""
}

output "service_account_email" {
  value = "\"${google_service_account.svision_service_acct.email}\""
}