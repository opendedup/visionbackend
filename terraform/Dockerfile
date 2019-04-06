FROM ubuntu:18.04

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y wget unzip \
    openssh-client \
    build-essential \
    golang \
    python3 \
    curl &&\
    apt-get clean && \
	  apt-get autoremove && \
	  rm -rf /var/lib/apt/lists/*
RUN rm -rf /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN gcloud config set core/disable_usage_reporting true && \
    gcloud config set component_manager/disable_update_check true && \
    gcloud config set metrics/environment github_docker_image && \
    gcloud --version


WORKDIR /svsetup
RUN apt update -y && \
    apt install -y wget unzip openssh-client build-essential golang
RUN wget https://releases.hashicorp.com/terraform/0.11.13/terraform_0.11.13_linux_amd64.zip
RUN unzip terraform_0.11.13_linux_amd64.zip && \
    rm terraform_0.11.13_linux_amd64.zip
ENV TERRAFORM_PATH=/svsetup/terraform
COPY deploy.py /svsetup
COPY runme.sh /svsetup
RUN chmod +x runme.sh
COPY cloud_setup /svsetup/cloud_setup
COPY local_setup /svsetup/local_setup


CMD [ "./runme.sh" ]
