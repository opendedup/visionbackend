# Suttle Vision
![alt text](https://github.com/suttlevision/svbackend/raw/master/githubui/svlogo.jpg "Suttle Vision Logo")

Suttlevision is an object detection pipeline for training and predicting objects within images for industrial applications. It includes 3 major components:
* Image Aquisition and Predition API Server (Capture)
* Model Preparation, Training and Preparation API Server (Pipeline API)
* Training and Image Prep Workers

## Setup Instructions

### Docker Pipeline Setup

1. Build the pipeline api server
```console
docker build -t pipeline-api -t pipeline-api:0.2  -f dockerimages/Docker.pipeline.api .
```

2. Build the Image aquisition server
```console
docker build -t capture -t capture:0.2  -f dockerimages/Docker.capture .
```
3. Build the Training and Prep Workers
```console
docker build -t train -t train:0.2  -f dockerimages/Docker.train.worker .
docker build -t prep -t prep:0.2  -f dockerimages/Docker.prep.worker .
```
4. Start Redis Docker container
```console
docker run --name some-redis -p 0.0.0.0:6379:6379 -d redis
```
5. Start Minio Storage container
```console
docker run -d -p 9000:9000 --name minio1 \
  -e "MINIO_ACCESS_KEY=imagerie" \
  -e "MINIO_SECRET_KEY=imagerie" \
  minio/minio server /data
```

6. Log into the minio ui console at http://localhost:9000 and create a bucket called imagerie0. If localhost does not work run `docker-machine ip default` and use that IP.
7. Start the Image aquisition server
```console
docker run -e BUCKET=imagerie0 -e ACCESS_KEY=imagerie -e SECRET_KEY=imagerie -d -e S3_URL=http://minio1:9000 --link=minio1 -p 0.0.0.0:5000:5000 capture
```
To start the image aquisition server mapping to a video port use:
```console
docker run -e BUCKET=imagerie0 -e ACCESS_KEY=imagerie -e SECRET_KEY=imagerie -d -e S3_URL=http://minio1:9000 --device=/dev/video0:/dev/video0 --link=minio1 -p 0.0.0.0:5000:5000 capture
```
8. Start the Pipeline API Server
```console
docker run -e REDIS_SERVER=some-redis -e REDIS_PORT=6379 -e ACCESS_KEY=imagerie -e SECRET_KEY=imagerie -d -e S3_URL=http://minio1:9000 --link=minio1 -p 0.0.0.0:5001:5000 --link some-redis pipeline-api
```
9. View the online documetation for the rest api's at
http://localhost:5000/api/
http://localhost:5001/api/

![alt text](https://github.com/suttlevision/svbackend/raw/master/githubui/screenshotapi.png "Logo Title Text 1")
