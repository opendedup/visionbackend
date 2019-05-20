variable "project_id" {
  type = "string"
  description = "The project id"
}

variable "bucket_name" {
  type = "string"
  description = "the gcp bucket where the data will be written"
}


variable "service_account_name" {
  type = "string"
  description = "the gcp service account name"
}

variable "service_account_email" {
  type = "string"
  description = "the gcp service account email address"
}

variable "instance_ip" {
  type = "string"
  description = "The cloud instance ip address"
}

variable "instance_name" {
  type = "string"
  description = "The cloud instance name"
}

variable "instance_zone" {
  type = "string"
  description = "The cloud instance zone location"
}

variable "jwt_token" {
  type = "string"
  description = "The jwt encryption token"
}

variable "cloud_password" {
  type = "string"
  description = "The cloud authentication password"
}

variable "location" {
  type = "string"
  description = "The region for the GCP project (Defaults to us-west1)"
  default = "us-west1"
}

resource "random_id" "instance_id" {
 byte_length = 8
}


resource "google_service_account_key" "fvision_key" {
  service_account_id = "${var.service_account_name}"
}

resource "google_compute_disk" "data_disk" {
  project = "${var.project_id}"
  name  = "fv-data-disk-${random_id.instance_id.hex}"
  type  = "pd-ssd"
  size = 50
  zone  = "${var.location}"
  labels = {
    environment = "flexiblevision"
  }
  physical_block_size_bytes = 4096
}

resource "google_compute_instance" "default" {
  project = "${var.project_id}"
  name         = "fvpredict-${random_id.instance_id.hex}"
  machine_type = "n1-standard-8"
  zone         = "${var.location}"
  allow_stopping_for_update = true
  tags = ["predict", "fvision-frontend"]

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "terminate"
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-1804-lts"
      size = 50
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type = "nvidia-tesla-t4"
    count = 1
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

  

  

  




  provisioner "remote-exec" {
    inline = [
      "mkdir -p  /home/ubuntu/fv_do_not_delete/"
      ]
    connection {
      type= "ssh"
      user="ubuntu"
      private_key="${file("${path.module}/id_rsa")}"
    }
    
  }
  provisioner "file" {
    content = "${base64decode(google_service_account_key.fvision_key.private_key)}"
    destination = "/home/ubuntu/fv_do_not_delete/fvision_creds.json"
    connection {
      type= "ssh"
      user="ubuntu"
      private_key="${file("${path.module}/id_rsa")}"
    }
  }

  provisioner "file" {
    source      = "${path.module}/resources/local_setup.sh"
    destination = "/tmp/local_setup.sh"
    connection {
      type= "ssh"
      user="ubuntu"
      private_key="${file("${path.module}/id_rsa")}"
    }
  }

  

  provisioner "remote-exec" {
    inline = [
      "chmod +x /tmp/local_setup.sh",
      " sudo /tmp/local_setup.sh ubuntu ${var.project_id} ${var.instance_name} ${var.instance_zone} ${var.bucket_name} ${var.jwt_token} ${var.cloud_password} latest" ,
      "rm /tmp/local_setup.sh"
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

resource "google_compute_firewall" "fvision_firewall" {
  project = "${var.project_id}"
  name = "fvision-firewall-predict"
  network = "default"
  allow {
    protocol = "tcp"
    ports = ["80"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags = ["fvision-frontend"]
}


output "instance_ip" {
 value = "\"${google_compute_instance.default.network_interface.0.access_config.0.nat_ip}\""
}

output "instance_url" {
 value = "\"http://${google_compute_instance.default.network_interface.0.access_config.0.nat_ip}\""
}

output "instance_name" {
 value = "\"${google_compute_instance.default.name}\""
}



