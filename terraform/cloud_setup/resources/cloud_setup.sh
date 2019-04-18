sudo apt update -y
sudo apt upgrade -y
sudo apt remove -y docker docker-engine docker.io containerd runc
sudo parted -s -a optimal /dev/disk/by-id/google-data mklabel gpt -- mkpart primary xfs 0% 100%
sudo sync /dev/disk/by-id/google-data
sleep 10
sudo mkfs.xfs /dev/disk/by-id/google-data-part1
sleep 10
sudo mkdir -p /var/lib/docker
echo "/dev/disk/by-id/google-data-part1 /var/lib/docker xfs defaults 0 2" | sudo tee --append /etc/fstab
sudo mount /var/lib/docker
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update -y
sudo apt install -y docker-ce docker-ce-cli
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo docker network create -d bridge imagerie_nw
sudo docker run -d --restart unless-stopped --name gcs-s3 --network imagerie_nw \
    -v /home/ubuntu/fvision_creds.json:/credentials.json -e "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" \
    -e "MINIO_ACCESS_KEY=imagerie" -e "MINIO_SECRET_KEY=imagerie" -e MINIO_CACHE_DRIVES=/cache -e MINIO_CACHE_EXPIRY=90 \
    -e MINIO_CACHE_MAXUSE=80 minio/minio gateway gcs $1
sudo docker run -d --restart unless-stopped --network imagerie_nw --name some-redis -d redis
sudo docker run -d --restart unless-stopped --network imagerie_nw --name prep -e REDIS_URL=redis://some-redis:6379 \
    -e ACCESS_KEY=imagerie -e SECRET_KEY=imagerie -e S3_URL=http://gcs-s3:9000 -v /home/ubuntu/fvision_creds.json:/credentials.json \
    -e "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" flexiblevision/prep
sudo docker run --restart unless-stopped -d --network imagerie_nw --name train -e REDIS_URL=redis://some-redis:6379 \
    -e ACCESS_KEY=imagerie -e SECRET_KEY=imagerie -e S3_URL=http://gcs-s3:9000 -v /home/ubuntu/fvision_creds.json:/credentials.json \
    -e "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" flexiblevision/train
sudo docker run --restart unless-stopped -d -p 0.0.0.0:80:5000 --name pipeline-api --network=imagerie_nw -e REDIS_SERVER=some-redis \
    -e BUCKET=$2 -e ACCESS_KEY=imagerie -e JWT_SECRET_KEY=$4 -e PASSWORD=$3 -e SECRET_KEY=imagerie -e S3_URL=http://gcs-s3:9000 \
    -v /home/ubuntu/fvision_creds.json:/credentials.json -e "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json" flexiblevision/pipeline-api
sudo docker run --restart unless-stopped -d --network imagerie_nw -p 0.0.0.0:5672:5672 --hostname my-rabbit -e RABBITMQ_DEFAULT_USER=user \
    -e RABBITMQ_DEFAULT_PASS=$3 --name some-rabbit rabbitmq:3
