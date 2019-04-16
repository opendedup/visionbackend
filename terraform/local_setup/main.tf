variable "camera_server" {
  type = "string"
  description = "the host name or ip address of the server where the cameras are attached"
}

variable "ssh_username" {
  type = "string"
  description = "the ssh username of the server where the cameras are attached"
}

variable "ssh_password" {
  type = "string"
  description = "the ssh password of the server where the cameras are attached"
}


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



resource "google_service_account_key" "fvision_key" {
  service_account_id = "${var.service_account_name}"
}

resource "null_resource" "local_server" {

  provisioner "remote-exec" {
    inline = [
      "mkdir -p ${var.ssh_username == "root" ? "/root/fv_do_not_delete/" : "/home/${var.ssh_username}/fv_do_not_delete/"}"
      ]
    connection {
      type= "ssh"
      user="${var.ssh_username}"
      password="${var.ssh_password}"
      host="${var.camera_server}"
    }
    
  }
  provisioner "file" {
    content = "${base64decode(google_service_account_key.fvision_key.private_key)}"
    destination = "${var.ssh_username == "root" ? "/root/fv_do_not_delete/fvision_creds.json" : "/home/${var.ssh_username}/fv_do_not_delete/fvision_creds.json"}"
    connection {
      type= "ssh"
      user="${var.ssh_username}"
      password="${var.ssh_password}"
      host="${var.camera_server}"
    }
  }

  provisioner "file" {
    source      = "${path.module}/resources/local_setup.sh"
    destination = "/tmp/local_setup.sh"
    connection {
      type= "ssh"
      user="${var.ssh_username}"
      password="${var.ssh_password}"
      host="${var.camera_server}"
    }
  }

  

  provisioner "remote-exec" {
    inline = [
      "chmod +x /tmp/local_setup.sh",
      "echo ${var.ssh_password} | sudo -S /tmp/local_setup.sh ${var.ssh_password} ${var.project_id} ${var.instance_name} ${var.instance_zone} ${var.bucket_name}",
      "rm /tmp/local_setup.sh"
      ]
    connection {
      type= "ssh"
      user="${var.ssh_username}"
      password="${var.ssh_password}"
      host="${var.camera_server}"
    }
  }

  provisioner "file" {
    source      = "${path.module}/resources/local_destroy.sh"
    destination = "/tmp/local_destroy.sh"
    when = "destroy"
    connection {
      type= "ssh"
      user="${var.ssh_username}"
      password="${var.ssh_password}"
      host="${var.camera_server}"
    }
  }

  provisioner "remote-exec" {
    when = "destroy"
    inline = [
      "chmod +x /tmp/local_destroy.sh",
      "echo ${var.ssh_password} | sudo -S /tmp/local_destroy.sh",
      "rm /tmp/local_destroy.sh"
      ]
    connection {
      type= "ssh"
      user="${var.ssh_username}"
      password="${var.ssh_password}"
      host="${var.camera_server}"
    }
  }

  

}



