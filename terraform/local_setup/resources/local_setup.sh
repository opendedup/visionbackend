apt update -y
apt upgrade -y
apt remove -y docker docker-engine docker.io containerd runc
apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
apt update -y
apt install -y docker-ce docker-ce-cli
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt update
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
usermod -aG docker $USER
systemctl enable docker
docker network create -d bridge imagerie_nw
docker run -d --restart unless-stopped --name gcs-s3 --network imagerie_nw \
    -v $HOME/sv_do_not_delete/svision_creds.json:/credentials.json -e "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" \
    -e "MINIO_ACCESS_KEY=imagerie" -e "MINIO_SECRET_KEY=imagerie" -e MINIO_CACHE_DRIVES=/cache -e MINIO_CACHE_EXPIRY=90 \
    -e MINIO_CACHE_MAXUSE=80 minio/minio gateway gcs $2
docker run -d --name=capdev -p 0.0.0.0:5000:5000 --restart unless-stopped --privileged -v /dev:/dev -v /sys:/sys -e BUCKET=$4 \
    --network imagerie_nw -e ACCESS_KEY=imagerie -e SECRET_KEY=imagerie -e S3_URL=http://gcs-s3:9000 -v $HOME/sv_do_not_delete/svision_creds.json:/credentials.json \
    -d -e "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" flexiblevision/capture:9e2990d
docker run -p 0.0.0.0:80:3000 --restart unless-stopped --name captureui -e CAPTURE_SERVER=http://capdev:5000 -d --network imagerie_nw -e PROCESS_SERVER=http://$3 flexiblevision/capture-ui
docker run -p 8500:8500 -p 8501:8501 --runtime=nvidia --name localprediction  -d -e AWS_ACCESS_KEY_ID=imagerie -e AWS_SECRET_ACCESS_KEY=imagerie -e AWS_REGION=us-east-1 -e S3_ENDPOINT=gcs-s3:9000 \
    --restart unless-stopped --network imagerie_nw -e S3_USE_HTTPS=0 \
    -t tensorflow/serving:1.12.0-gpu --model_config_file=s3://$4/trained_models/model.config


