import logging

from flask import request
from flask_restplus import Resource, Namespace, fields

import boto3
import numpy as np
import uuid
import json

from pathlib import Path

import datetime
import os
import traceback

from flask import jsonify
from flask import make_response
from flask import g
from flask import request
from flask import Response

from flask_jwt_extended import jwt_required, get_jwt_identity

import concurrent.futures
import multiprocessing

import settings
import threading

log = logging.getLogger(__name__)

global_lock = threading.Lock()


ns = Namespace('models', description='Manage Prediction Models')

category = ns.model('Category', {
    'id': fields.Integer(required=True, description='The index id for the tag',
                          example=1),
    'name': fields.String(required=True, description='The category name',
                          example='Pins'),
})

mversion = ns.model('ModelVersion', {
    'model_path': fields.String(required=True, description='The model path',
                          example='trained_models/model0'),
    'model_version': fields.Integer(required=True, description='The model version',
                          example=1545285807342),
    'test_samples': fields.Integer(required=True, description='The number of samples used for testing the model',
                          example=100),
    'categories':fields.List(fields.Nested(category,required=True, description='The object tags for this training')),

    'train_samples': fields.Integer(required=True, description='The number of samples used for training the model',
                          example=1000),
    'training':fields.String(required=True, description='The corpus used to train the model',
                          example='data1'),
    'name': fields.String(required=True, description='The model name',
                          example='model0'),
    'project_name': fields.String(required=True, description='The name of the project assigned to this training',
                          example='project0'),
})

model = ns.model('Model', {
    'name': fields.String(required=True, description='The model name',
                          example='model0'),
    'project_name': fields.String(required=True, description='The name of the project assigned to this training',
                          example='project0'),
    'model_path': fields.String(required=True, description='The model path',
                          example='trained_models/model0'),
    'label_file': fields.String(required=True, description='The model path',
                          example='trained_models/model0/object-detection.pbtxt'),
    'versions' : fields.List(fields.Nested(mversion,required=False, description='Model Versions Available')),
    'bucket':fields.String(required=True, description='The bucket where the data is located',
                          example='bucket0'),
})

def download_jobdef(fldr,file,mp):
    tl = file[len(fldr):]
    t = tl.split('/')
    mm = None
    if tl.endswith('job_def.json'):
        jd = settings.storage.download_to_string(file)
        jdo = json.loads(jd)
        with global_lock:
            if t[0] not in mp:
                mm = {
                    'name' : t[0],
                    'project_name': jdo['project_name'],
                    'model_path':'{}{}/{}'.format(fldr,t[0],t[1]),
                    'versions': [jdo],
                    'label_file':'{}{}/{}/{}'.format(fldr,t[0],t[1],'object-detection.pbtxt'),
                    'bucket': settings.data['BUCKET']
                }
                mp[t[0]]=mm
            elif t[0] in mp and tl.endswith('job_def.json'):
                mm = mp[t[0]]
                jd = settings.storage.download_to_string(file)
                mm['versions'].append(jdo)
    return mm

def download_jobdef_proj(fldr,file,mm):

    tl = file[len(fldr):]
    t = file.split('/')
    if tl.endswith('job_def.json'):
        jd = settings.storage.download_to_string(file)
        jdo = json.loads(jd)
        with global_lock:
            if 'name' not in mm:
                mm['name'] = t[0]
                mm['project_name'] =jdo['project_name']
                mm['model_path'] = '{}{}/{}'.format(fldr,t[0],t[1])
                mm['versions']= [jdo]
                mm['label_file'] = '{}{}/{}/{}'.format(fldr,t[0],t[1],'object-detection.pbtxt')
                mm['bucket'] = settings.data['BUCKET']
            else:
                mm['versions'].append(jdo)
    return mm

@ns.route('/')
class ModelCollection(Resource):
    @ns.marshal_list_with(model)
    @jwt_required
    def get(self):
        """
        Returns list of Models.
        """
        fldr = 'trained_models/'
        lst = settings.storage.list_files(fldr)
        mp = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
            future_img = {executor.submit(
                download_jobdef, fldr, l, mp): l for l in lst}
            for future in concurrent.futures.as_completed(future_img):
                #img = future_img[future]
                try:
                    future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' %
                          ("create_tf_example", exc))
                    var = traceback.format_exc()
                    print(var)
            
                
            
        return list(mp.values())


@ns.route('/<string:name>')
@ns.response(404, 'Model not found.')
class ModelItem(Resource):

    @ns.marshal_with(model)
    @jwt_required
    def get(self, name):
        """
        Returns a model.
        """
        fldr = 'trained_models/' + name + '/'
        lst = settings.storage.list_files(fldr)
        mm = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
            future_img = {executor.submit(
                download_jobdef_proj, fldr, l, mm): l for l in lst}
            for future in concurrent.futures.as_completed(future_img):
                #img = future_img[future]
                try:
                    future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' %
                          ("create_tf_example", exc))
                    var = traceback.format_exc()
                    print(var)
                    
                
        if mm == None:
            return 'Model not found.',404    
        else:
            return mm

@ns.route('/<string:project>/<string:model_id>')
@ns.param('project', 'The project name associated with the model id')
@ns.param('model_id', 'The model id to delete')
@ns.response(404, 'Model not found.')
class ModelVersionItem(Resource):

    @ns.response(204, 'Model successfully deleted.')
    @jwt_required
    def delete(self, project,model_id):
        """
        Deletes a model version.
        """
        if not settings.storage.exists("trained_models/{}/{}/job_def.json".format(project,model_id)):
            return 'model with that name does not exist',404
        else:
            settings.storage.delete_cloud_dir("trained_models/{}/{}/".format(project,model_id))
        return None, 204