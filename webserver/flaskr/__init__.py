import os

import boto3
from botocore.errorfactory import ClientError
import numpy as np
import uuid
import json

from pathlib import Path


import datetime

import logging.config

from flask import Flask, Blueprint
from flask import Flask, render_template, send_from_directory
from flask import jsonify
from flask import make_response
from flask import g
from flask import request
from flask import Response
from model.cameras import Cameras
from model.project import Project
from model.project import projects
from api.endpoints.predict import ns as predict_ns
from api.endpoints.train import ns as train_ns
from api.endpoints.projects import ns as project_ns
from api.endpoints.cameras import ns as cameras_ns
from api.endpoints.models import ns as models_ns
from api.endpoints.corpus import ns as corpus_ns
from api.endpoints.jobs import ns as jobs_ns
from utils.storage import Storage
from flask_restplus import Api
from flask_cors import CORS

if 'SPINNAKER' in os.environ and os.environ['SPINNAKER'].lower() =='true':
    import PySpin

import socket
import settings

def create_app(test_config=None):
    logging_conf_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../logging.config'))
    print(logging_conf_path)
    logging.config.fileConfig(logging_conf_path)
    log = logging.getLogger(__name__)

    app = Flask(__name__,instance_relative_config=True)
    CORS(app, resources={r"/api/*": {"origins": "*"}});
    
    app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP
    blueprint = Blueprint('api', __name__, url_prefix='/api/capture')
    settings.api = Api(blueprint,version='1.0', title='Flexible Vision Capture and Detection API',
              description='An Image capture and object detection api')
    
    print(settings.api)
    settings.api.add_namespace(project_ns)
    settings.api.add_namespace(train_ns)
    settings.api.add_namespace(predict_ns)
    settings.api.add_namespace(cameras_ns)
    settings.api.add_namespace(models_ns)
    settings.api.add_namespace(corpus_ns)
    settings.api.add_namespace(jobs_ns)
    
    app.register_blueprint(blueprint)


    config_file = Path('config.json')
    if config_file.is_file():
        with open('config.json') as f:
            settings.data = json.load(f)
    else:
        if 'XRES' in os.environ:
            settings.data['xres'] = int(os.environ["XRES"])
        else:
            settings.data['xres'] = 640
        if 'WORKSTATION_NAME' in os.environ:
            settings.data['workstation_name'] = os.environ['WORKSTATION_NAME']
        else:
            settings.data['workstation_name'] = socket.gethostname()
        if 'YRES' in os.environ:
            settings.data['yres'] = int(os.environ["YRES"])
        else:
            settings.data['yres'] = 480
        if 'BW' in os.environ:
            settings.data['bw'] = True
        else:
            settings.data['bw'] = False
        if 'ACCESS_KEY' in os.environ:
            settings.data['ACCESS_KEY'] = os.environ['ACCESS_KEY']
        if 'SECRET_KEY' in os.environ:
            settings.data['SECRET_KEY'] = os.environ['SECRET_KEY']
        if 'S3_URL' in os.environ:
            settings.data['S3_URL'] = os.environ['S3_URL']
        if 'BUCKET' in os.environ:
            settings.data['BUCKET'] = os.environ['BUCKET']
        if 'USE_GCS' in os.environ and os.environ['USE_GCS'].lower() =='true':
            settings.data['USE_GCS'] = True
        if 'SPINNAKER' in os.environ and os.environ['SPINNAKER'].lower() =='true':
            settings.data['SPINNAKER'] = True
            settings.pgsystem = PySpin.System.GetInstance()
        else:
            settings.data['SPINNAKER'] = False
        settings.data['image_type'] = '.jpg'
    settings.storage = Storage(settings.data)
    settings.camera = Cameras(settings.data)
    #Load existing projects if you can
    if 'BUCKET' in settings.data and 'URL' in settings.data:
        statefile =  'projects/projects.json'
        try:
            jsonbody = settings.storge.download_to_string(statefile)
            state = json.loads(jsonbody)
            for key, value in state.items():
                projects[key] = Project(key,value)
            print(projects)
        except ClientError:
            pass
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass



    @app.route('/')
    def root():
        return render_template('index.html')
    
    @app.after_request
    def add_header(r):
        """
        Add headers to both force latest IE rendering engine or Chrome Frame,
        and also to cache the rendered page for 10 minutes.
        """
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        r.headers['Cache-Control'] = 'public, max-age=0'
        return r

    return app
