#!/bin/bash
if [ ! -f /fvsetup/cloud_setup/id_rsa ]; then
    ssh-keygen -b 2048 -t rsa -f /fvsetup/cloud_setup/id_rsa -q -N ""
fi
if [ ! -f /fvsetup/cloud_setup_predict/id_rsa ]; then
    ssh-keygen -b 2048 -t rsa -f /fvsetup/cloud_setup_predict/id_rsa -q -N ""
fi
python /fvsetup/deploy.py