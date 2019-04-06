#!/bin/bash
if [ ! -f /svsetup/cloud_setup/id_rsa ]; then
    ssh-keygen -b 2048 -t rsa -f /svsetup/cloud_setup/id_rsa -q -N ""
fi
python /svsetup/deploy.py