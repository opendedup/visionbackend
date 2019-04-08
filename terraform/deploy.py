import getpass
import sys
import subprocess
import os


terraform = os.environ["TERRAFORM_PATH"]

options = ['Deploy Cloud Components','Setup Local Camera Server',
    'Remove Cloud Components','Remove Local Camera Server','View Cloud Configuration','Login to Google Cloud',
    'Logout to Google Cloud','Exit']

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def remove_cloud_instance():
    check_gcp_login()
    p = query_yes_no("Remove the CLOUD INSTACE for flexible Vision ? ",default="no")
    if p:
        sys.stdout.write("######################      Warning     ########################\n")
        sys.stdout.write("THIS WILL REMOVE THE CLOUD INSTANCE AND ALL DATA.\n")
        p = query_yes_no("ARE YOU SURE YOU WANT TO DO THIS ? ",default="no")
        if p:
            subprocess.call([terraform,"init","cloud_setup/"])
            subprocess.call([terraform,"destroy","-var-file=cloud_config/input.tfvars","-state=cloud_config/terraform.tfstate","cloud_setup/"])
            f = open("cloud_config/output.tfvars", "w")
            subprocess.call([terraform,"output","-state=cloud_config/terraform.tfstate"],stdout=f)
            f.close()
            os.remove("cloud_config/output.tfvars")

def remove_local_instance():
    check_gcp_login()
    p = query_yes_no("Remove a LOCAL CAMERA SERVER for flexible Vision ? ",default="no")
    if p:
        sys.stdout.write("######################      Warning     ########################\n")
        sys.stdout.write("THIS WILL REMOVE A LOCAL CAMERA SERVER AND ALL CONFIG DATA ON IT.\n")
        sys.stdout.write("IT WILL NOT REMOVE CLOUD DATA.\n")
        p = query_yes_no("ARE YOU SURE YOU WANT TO DO THIS ? ",default="no")
        if p:
            sys.stdout.write("What is the IP Address or Host Name of the local camera server ? ")
            ip = input().lower()
            sys.stdout.write("What is the SSH username to attach to the local camera server ? ")
            user = input().lower()
            password = getpass.getpass("What is the SSH Password to access the local camera server ? ")
            with open ("cloud_config/output.tfvars", "r") as varfile:
                config = varfile.read()
            config += "camera_server = \"{}\"\n".format(ip)
            config += "ssh_username = \"{}\"\n".format(user)
            config += "ssh_password = \"{}\"\n".format(password)
            with open ("/tmp/input.tfvars", "w") as varfile:
                varfile.write(config)
            subprocess.call([terraform,"init","local_setup/"])
            subprocess.call([terraform,"destroy","-var-file=/tmp/input.tfvars","-state=local_config_{}/terraform.tfstate".format(ip),"local_setup/"])
            os.remove("/tmp/input.tfvars")        

def deploy_cloud_instance():
    check_gcp_login()
    p = query_yes_no("Setup a cloud instance of flexible Vision ? ")
    if p:
        sys.stdout.write("What GCP region would you like to deploy in [us-west1] ? ")

        region = input().lower()
        if len(region) == 0:
            region = "us-west1"
        sys.stdout.write("What GCP Organization would you like to deploy in [none] ? ")

        org = input().lower()
        sys.stdout.write("What is your billing account [My Billing Account] ? ")
        billing_acct = input()
        if len(billing_acct) == 0:
            billing_acct = "My Billing Account"
        subprocess.call(["mkdir","-p","cloud_config"])
        with open ("cloud_config/input.tfvars", "w") as varfile:
            varfile.write("location = \"{}\"\n".format(region))
            varfile.write("gcp_organization_id = \"{}\"\n".format(org))
            varfile.write("gcp_billing_account = \"{}\"\n".format(billing_acct))
        subprocess.call([terraform,"init","cloud_setup/"])
        subprocess.call([terraform,"apply","-var-file=cloud_config/input.tfvars","-auto-approve","-state=cloud_config/terraform.tfstate","cloud_setup/"])
        f = open("cloud_config/output.tfvars", "w")
        subprocess.call([terraform,"output","-state=cloud_config/terraform.tfstate"],stdout=f)
        f.close()

def deploy_local_cam():
    check_gcp_login()
    exists = os.path.isfile('cloud_config/input.tfvars')
    if not exists:
        sys.stdout.write("File cloud_config/output.tfvars does not exist.\n")
        sys.stdout.write("Deploy Cloud Components before continuing.\n")
    p = query_yes_no("Deploy a flexible Vision local camera server?")
    if p:
        sys.stdout.write("######################      Warning     ########################\n")
        sys.stdout.write("Local Camera Servers Must :\n")
        sys.stdout.write("\t1. Have A local NVIDIA Graphics Card\n")
        sys.stdout.write("\t2. Be Running Ubuntu 18.04\n")
        sys.stdout.write("\t3. Be Available through SSH\n")
        sys.stdout.write("\t4. Have a USB Camera Attached\n")
        p = query_yes_no("Does your local camera server meet these requirements ? ")
        if p:
            sys.stdout.write("What is the IP Address or Host Name of the local camera server ? ")
            ip = input().lower()
            sys.stdout.write("What is the SSH username to attach to the local camera server ? ")
            user = input().lower()
            password = getpass.getpass("What is the SSH Password to access the local camera server ? ")
            with open ("cloud_config/output.tfvars", "r") as varfile:
                config = varfile.read()
            config += "camera_server = \"{}\"\n".format(ip)
            config += "ssh_username = \"{}\"\n".format(user)
            config += "ssh_password = \"{}\"\n".format(password)
            with open ("/tmp/input.tfvars", "w") as varfile:
                varfile.write(config)
            subprocess.call([terraform,"init","local_setup/"])
            subprocess.call([terraform,"apply","-var-file=/tmp/input.tfvars","-auto-approve","-state=local_config_{}/terraform.tfstate".format(ip),"local_setup/"])
            os.remove("/tmp/input.tfvars")

def output_cloud_setup():
    subprocess.call([terraform,"output","-state=cloud_config/terraform.tfstate"])

def check_gcp_login():
    cmd = subprocess.Popen(['gcloud','auth','list','--filter','status:ACTIVE','--format','value(account)'], stdout=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    if len(cmd_out) ==0:
        subprocess.call(["gcloud","auth","login"])
        subprocess.call(["gcloud","auth","application-default","login"])
    else:
        sys.stdout.write("Logged in to GCP as {}\n".format(cmd_out.decode("utf-8")))

def revoke_gcp_login():
    cmd = subprocess.Popen(['gcloud','auth','revoke'], stdout=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    sys.stdout.write(cmd_out.decode("utf-8"))

def main():
    sys.stdout.write("Welcome to the flexible Vision Setup.\n")
    while True:
        for i in range(len(options)):
            sys.stdout.write("[{}] {}\n".format(i,options[i]))
        sys.stdout.write("Select a setup option : ")
        try:
            option = int(input())
            if option == 7:
                sys.stdout.write("Exiting\n")
                break
            if option == 0:
                deploy_cloud_instance()
            if option == 1:
                deploy_local_cam()
            if option == 2:
                remove_cloud_instance()
            if option == 3:
                remove_local_instance()
            if option == 4:
                output_cloud_setup()
            if option == 5:
                check_gcp_login()
            if option == 6:
                revoke_gcp_login()
        except ValueError:
            pass
        





    
if __name__ == '__main__':
    main()