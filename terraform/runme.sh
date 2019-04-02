#!/bin/bash
if [ ! -f /root/.ssh/id_rsa ]; then
    ssh-keygen -b 2048 -t rsa -f ~/.ssh/id_rsa -q -N ""
fi
python /svsetup/deploy.py