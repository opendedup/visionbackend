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
from controllers.train import train_job
from controllers.train import train_mlengine
from controllers.train import export_mlengine

from redis import Redis
from rq import Queue, get_failed_queue
from rq.registry import StartedJobRegistry
from rq.registry import BaseRegistry
from rq.registry import FinishedJobRegistry

from utils.storage import Storage


log = logging.getLogger(__name__)

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

print('redis={}{}'.format(redis_server,str(redis_port)))
redis_con=Redis(redis_server,redis_port,password=redis_password)
aug_queue = Queue('aug',connection=redis_con)
registry = StartedJobRegistry('aug', connection=redis_con)
fregistry = FinishedJobRegistry('aug', connection=redis_con)
bregistry = BaseRegistry('aug', connection=redis_con)
ns = Namespace('prep', description='Prepare a vision corpus for training')


aug_rules = ns.model('Augmentation_Rules', {
    'remove_out_of_image': fields.String(required=False,description="""Specify "cut_out_partly" to cut out objects that are partly out of view.
    Specify "cut_out_of_image" to trim the box to the image field of view. Specify "leave_partly_in_image" to keep all boxes regardless of their field of view.
     """,example="cut_out_of_image",
     enum=['leave_partly_in_image',
                                 'cut_out_partly','cut_out_of_image'],
                              default="cut_out_of_image"),
    'hflip':fields.Float(required=False, description='Horizontally Flip and Image a designated percentage of the time.',
                          default=0,
                          example=.5),
    'vflip':fields.Float(required=False, description='Vertically Flip and Image a designated percentage of the time.',
                          default=0,
                          example=.5),
    'rotate':fields.List(fields.Integer,required=False,description='Rotate objects clockwise and counterclockwise',
                          max_items=2,min_items=2,example=[90,-90]),
    'scale':fields.Boolean(required=False,description='Randomly add slightly scale the image randomly between 0.8x - 1x. ',
                              default=False),
    'shear':fields.Boolean(required=False,description='Randomly Slightly shear the image. ',
                              default=False),
    'contrast':fields.Boolean(required=False,description='Randomly add contrast to image. ',
                              default=False),
    'noise':fields.Boolean(required=False,description='Randomly add noise to images. ',
                              default=False),
    'crop':fields.Boolean(required=False,description='Randomly crop images. ',
                              default=False)
})

prep_job = ns.model('Prep_Job', {
    'name': fields.String(required=True,
                          description='The name of the prep corpus that will be returned from the prep job',
                          example='data0'),
    'bucket':fields.String(required=False, description='The s3 bucket target',
                           example='bucket0'),
    'aug':fields.Boolean(required=False,description='If true data will be augmented as part of prep. ',
                              default=True),
    'train_samples':fields.Integer(required=False, description='The number augemented training samples to create per image.',
                          default=1000,
                          example=1000),
    'test_samples':fields.Integer(required=False, description='The number augemented testing samples to create per image.'+
                                  ' Testing samples should be 20% of training samples.',
                          default=200,
                          example=200),
    'project_name': fields.String(required=True,
                          description='The name of the project to us as the corpus source.',
                          example='project0'),
    'test_percentage' : fields.Float(required=False,
                          description='The percentage of the corpus to use for test data.',
                          example=0.2, default=0.0),
    'use_gcs':fields.Boolean(required=False,description='Use GCS for storage.',
                              default=False),
    'aug_rules': fields.Nested(aug_rules,required=False, description='Image Augmentation rules used for image augmentation'),
    'desired_size':fields.Integer(required=False, description='The disired square dimensions with padding in pixels. If set to 0, this is ignored',
                          default=0,
                          example=640),
    
})



@ns.route('/run')
class RunPrep(Resource):
    @ns.response(201, '{"status":"queued","job_id":"uuid"}')
    @ns.expect(prep_job)
    def post(self):
        """
        Executes a prep job to create an image corpus for training.
        Use this method to start a prep job.
        """
        job_def = request.json
        job_def['process_json'] = True # Hardcode to process json file from project folder
        job = Job(job_def['name'],job_def)
        job.type = 'preprocess'
        dt = newdt.now()
        job.start_time = int(dt.timestamp()*1000)
        job.request = {'full_path': request.full_path,'remote_addr':request.remote_addr,'method':request.method}
        jb = aug_queue.enqueue(
             preprocess, job,timeout=-1,result_ttl=86400,ttl=-1)
        jb.meta['job_def'] = job_def
        dt = newdt.now()
        jb.meta['job_init_time'] = str(int(dt.timestamp()*1000))

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
        storage.upload_data(json_str,'jobs/running/{}_0_preprocess_r_{}.json'.format(str(job.start_time),jb.id),contentType='application/json')
        storage.upload_data(json_str,'jobs/all/{}_0_preprocess_r_{}.json'.format(str(job.start_time),jb.id),contentType='application/json')

        return {
            "status": jb.status,
            'job_id': jb.id,
            'meta':jb.meta},201



@ns.route('/job/<string:id>')
class PrepJob(Resource):
    @ns.response(200, 'Returns a job status from queue')
    def get(self,id):
        """
        Returns job metadata for a given id.
        """
        jb = None
        jb =aug_queue.fetch_job(id)
        if jb is None:
            resp = {'no_such_job_id': id}
            return resp, 404
        else:
            return {
            "status": jb.status,
            'job_id': jb.id,
            'meta':jb.meta},200


@ns.route('/jobs/running')
class PrepRunningJobs(Resource):
    @ns.response(200, 'Returns running jobs from the queue')
    def get(self):
        """
        Returns a list of running jobs.
        """
        job_ids = registry.get_job_ids()
        print(job_ids)
        ar = []
        for id in job_ids:
            job = None
            job =aug_queue.fetch_job(id)
            if job is not None:
                jb = {
                    "status": job.status,
                    'job_id': job.id,
                    'meta':job.meta}
                ar.append(jb)
        return ar,200

@ns.route('/jobs/failed')
class PrepFailedJobs(Resource):
    @ns.response(200, 'Returns failed jobs from the queue')
    def get(self):
        """
        Returns a list of failed jobs.
        """
        job_ids = get_failed_queue(redis_con).jobs
        ar = []
        for job in job_ids:
            jb = {
            "status": job.status,
            'job_id': job.id,
            'meta':job.meta}
            ar.append(jb)
        return ar,200

@ns.route('/jobs/finished')
class PrepFinishedJobs(Resource):
    @ns.response(200, 'Returns finished jobs from the queue')
    def get(self):
        job_ids = fregistry.get_job_ids()
        return job_ids,200
