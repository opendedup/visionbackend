docker stop gcs-s3 capdev captureui localprediction
docker rm gcs-s3 capdev captureui localprediction
docker image rm flexiblevision/capture-ui
docker image rm minio/minio
docker image rm flexiblevision/capture:9e2990d
docker image rm tensorflow/serving:1.12.0-gpu
docker network rm imagerie_nw


