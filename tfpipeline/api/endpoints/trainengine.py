import logging

from flask import request
from flask_restplus import Resource, Namespace, fields

import boto3
import numpy as np
import json

from pathlib import Path

from datetime import datetime as newdt
import os

from flask import make_response
from flask import g
from flask import request
from flask import Response

from models.job import Job
from controllers.preprocess import preprocess
from controllers.train import train_job as train_job_method
from controllers.train import train_mlengine
from controllers.train import export_mlengine

from flask_jwt_extended import jwt_required, get_jwt_identity

from redis import Redis
from rq import Queue
from rq.registry import StartedJobRegistry
from rq.registry import FinishedJobRegistry

log = logging.getLogger(__name__)

from utils.storage import Storage

redis_server = 'localhost'
redis_port = 6379
redis_password = None
access_key = ''
secret_key=''
s3_url = ''
if 'REDIS_SERVER' in os.environ:
    redis_server = os.environ["REDIS_SERVER"]
if 'REDIS_PORT' in os.environ:
    redis_port = int(os.environ["REDIS_PORT"])
if 'REDIS_PASSWORD' in os.environ:
    redis_password = os.environ["REDIS_PASSWORD"]
if 'ACCESS_KEY' in os.environ:
    access_key = os.environ["ACCESS_KEY"]
if 'SECRET_KEY' in os.environ:
    access_key = os.environ["SECRET_KEY"]
if 'S3_URL' in os.environ:
    s3_url = os.environ["S3_URL"]

redis_con=Redis(redis_server,redis_port,password=redis_password)
train_queue = Queue('train',connection=redis_con)
registry = StartedJobRegistry('train', connection=redis_con)
fregistry = FinishedJobRegistry('train', connection=redis_con)
flregistry = train_queue.failed_job_registry
ns = Namespace('train', description='Train a vision corpus on a prepared dataset')

train_job = ns.model('Train_Job', {
    'name': fields.String(required=True,
                          description='The name of the model that will be trained',
                          example='project0'),
    'bucket':fields.String(required=False, description='The s3 bucket target',
                           example='bucket0'),
    'model':fields.String(required=True, description='The pretrained model to start with',
                           example='faster_rcnn_inception_v2_coco',
                           enum=['faster_rcnn_inception_v2_coco',
                                 'ssd_resnet50_v1_fpn_shared_box_predictor_coco14_sync',
                                 'ssd_mobilenet_v1_fpn_shared_box_predictor_coco14_sync',
                                 'faster_rcnn_resnet101_coco','faster_rcnn_inception_resnet_v2_atrous_coco',
                                 'faster_rcnn_resnet50_coco']),
    'num_train_steps':fields.Integer(required=True, description='The number of steps to train',
                             example=50000),
    'parameter_servers':fields.Integer(required=True, description='The number of parameter servers to bring'+
                            ' up with the ml engine model. If 0 specified it will automatically create the correct number.',
                             example=0,default=0),
    'prep_name':fields.String(required=True, description='The corpus to train against',
                          example='data0'),
    'max_dim':fields.Integer(required=True, description='The maximum length of the largest' +
                            ' dimension of the image. If the image\'s width/heigh is larger than this' +
                            ' dimension is will be resized to max_dim' ,
                          example=1280),
    'min_dim':fields.Integer(required=True, description='The maximum length of the smallest' +
                            ' dimension of the image. If the image\'s width/heigh is larger than this' +
                            ' dimension is will be resized to min_dim',
                          example=720),
    'batch_size':fields.Integer(required=True, description='The number of images to process in training in one pass.'+
                          ' A larger number is increases speed at the expense of memory.',
                          default=2,
                          example=2),
    'ml_engine':fields.Boolean(required=False,description='If true use mlengine ' +
                              'for training',
                              default=True),
    'ml_workers':fields.Integer(required=False, description='The number of workers to use for an ml engine training',
                          default=8,
                          example=8),
    'use_gcs':fields.Boolean(required=False,description='Use GCS for storage.',
                              default=False),
    'use_tpu':fields.Boolean(required=False,description='Use a Google Tensor Processing Unit (TPU).',
                              default=False),
    'source_training':fields.String(required=False, description='A previous training to use as the basis for this training.',
                          example='training0', default=''),
    'mle_region':fields.String(required=False, description='The region to perform ml engine trainings in.',
                          example='us-central1', default='us-central1')
})

export_job = ns.model('Export_Job', {
    'project_name': fields.String(required=True,
                          description='The name of the project that the model was trained against',
                          example='project0'),
    'bucket':fields.String(required=False, description='The s3 bucket target',
                           example='bucket0'),
    'training':fields.String(required=True, description='The training job to start with',
                           example='training0',
                           ),
    'deploy_ml_engine':fields.Boolean(required=False,description='Deploy in Google ML Engine for prediction.',
                              default=False),
    'use_gcs':fields.Boolean(required=False,description='Use GCS for storage.',
                              default=False),
    'model_name':fields.String(required=True,description='Use GCS for storage.',
                              example='model3'),
})

@ns.route('/run')
class RunTraining(Resource):
    @ns.response(201, '{"status":"queued","job_id":"uuid"}')
    @ns.expect(train_job)
    @jwt_required
    def post(self):
        """
        Executes a training.
        Use this method to start a training.
        """
        job_def = request.json
        job = Job(job_def['name'],job_def)
        job.type = 'train'
        dt = newdt.now()
        job.start_time = int(dt.timestamp()*1000)
        job.request = {'full_path': request.full_path,'remote_addr':request.remote_addr,'method':request.method}
        if hasattr(job,'ml_engine') and job.ml_engine:
            jb = train_queue.enqueue(
                 train_mlengine, job,timeout=-1,result_ttl=-1)
        else:
            jb = train_queue.enqueue(
             train_job_method, job,timeout=-1)
        jb.meta['job_init_time'] = str(int(dt.timestamp()*1000))
        
        jb.meta['job_def'] = job_def
        jb.save_meta()
        
        json_str = job.to_json_string()
        st = {
            'BUCKET' : job.bucket,
            'USE_GCS' : job.use_gcs,
            'ACCESS_KEY' : access_key,
            'SECRET_KEY' : secret_key,
            'S3_URL' : s3_url
        }
        storage = Storage(st)
        storage.upload_data(json_str,'jobs/running/{}_0_train_r_{}.json'.format(str(job.start_time),jb.id),contentType='application/json')
        storage.upload_data(json_str,'jobs/all/{}_0_train_r_{}.json'.format(str(job.start_time),jb.id),contentType='application/json')
        return {
            "status": jb.status,
            'job_id': jb.id,
            'meta':jb.meta},201
        
@ns.route('/export')
class ExportTraining(Resource):
    @ns.response(201, '{"status":"queued","job_id":"uuid"}')
    @ns.expect(export_job)
    @jwt_required
    def post(self):
        """
        Exports a finished training for prediction.
        Use this method to export a finished training job for prediction.
        """
        job_def = request.json
        job = Job(job_def['project_name'],job_def)
        job.type = 'export'
        job.request = {'full_path': request.full_path,'remote_addr':request.remote_addr,'method':request.method}
        jb = train_queue.enqueue(export_mlengine, job,timeout=-1,result_ttl=-1)
        dt = newdt.now()
        job.start_time = int(dt.timestamp()*1000)
        jb.meta['job_init_time'] = str(int(dt.timestamp()*1000))
        jb.meta['job_def'] = job_def
        jb.save_meta()
        json_str = job.to_json_string()
        st = {
            'BUCKET' : job.bucket,
            'USE_GCS' : job.use_gcs,
            'ACCESS_KEY' : access_key,
            'SECRET_KEY' : secret_key,
            'S3_URL' : s3_url
        }
        storage = Storage(st)
        storage.upload_data(json_str,'jobs/running/{}_0_export_r_{}.json'.format(str(job.start_time),jb.id),contentType='application/json')
        storage.upload_data(json_str,'jobs/all/{}_0_export_r_{}.json'.format(str(job.start_time),jb.id),contentType='application/json')

        return {
            "status": jb.status,
            'job_id': jb.id,
            'meta':jb.meta},201

@ns.route('/job/<string:id>')
class TrainQJob(Resource):
    @ns.response(200, 'Returns a job status from queue')
    @jwt_required
    def get(self,id):
        jb = None
        jb =train_queue.fetch_job(id)
        if jb is None:
            resp = {'no_such_job_id': id}
            return resp, 404
        else:
            return {
            "status": jb.status,
            'job_id': jb.id,
            'meta':jb.meta},200

@ns.route('/jobs/running')
class TrainRunningJobs(Resource):
    @ns.response(200, 'Returns runing jobs from the queue')
    @jwt_required
    def get(self):
        """
        Returns a list of running jobs.
        """
        job_ids = registry.get_job_ids()
        print(job_ids)
        ar = []
        for id in job_ids:
            job = None
            job =train_queue.fetch_job(id)
            if job is not None:
                jb = {
                    "status": job.status,
                    'job_id': job.id,
                    'meta':job.meta}
                ar.append(jb)
        return ar,200

@ns.route('/jobs/failed')
class TrainFailedJobs(Resource):
    @ns.response(200, 'Returns failed jobs from the queue')
    @jwt_required
    def get(self):
        """
        Returns a list of failed jobs.
        """
        job_ids = flregistry.get_job_ids()
        return job_ids,200

@ns.route('/jobs/finished')
class TrainFinishedJobs(Resource):
    @ns.response(200, 'Returns finished jobs from the queue')
    @jwt_required
    def get(self):
        current_user = get_jwt_identity()
        job_ids = fregistry.get_job_ids()

        return job_ids,200
