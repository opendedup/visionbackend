#!/bin/bash
VERSION=0.1
docker build -t flexiblevision/base -t flexiblevision/base:$VERSION  -f dockerimages/Docker.base .
docker push flexiblevision/base
docker build -t flexiblevision/pipeline-api -t flexiblevision/pipeline-api:$VERSION  -f dockerimages/Docker.pipeline.api .
docker push flexiblevision/pipeline-api
docker build -t flexiblevision/train -t flexiblevision/train:$VERSION  -f dockerimages/Docker.train.worker .
docker push flexiblevision/train
docker build -t flexiblevision/prep -t flexiblevision/prep:$VERSION  -f dockerimages/Docker.prep.worker .
docker push flexiblevision/prep
docker build  -t flexiblevision/capture -t flexiblevision/capture:$VERSION  -f dockerimages/Docker.capture .
docker push flexiblevision/capture
docker build -t flexiblevision/capture-pointgrey:latest -t flexiblevision/capture-pointgrey:$VERSION  -f dockerimages/Docker.pg.capture .
docker push flexiblevision/capture-pointgrey:latest
docker build -t flexiblevision/setup:latest -t flexiblevision/setup:$VERSION  -f terraform/Dockerfile .
docker push flexiblevision/setup