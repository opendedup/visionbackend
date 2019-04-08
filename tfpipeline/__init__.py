import os

import boto3
import numpy as np
import json

from pathlib import Path


import datetime

import settings

from datetime import datetime as newdt

from flask import Flask, Blueprint
from flask import Flask, render_template, send_from_directory
from flask import jsonify
from flask import g
from flask import request
from flask import Response

from flask_restplus import Api

from api.endpoints.prepengine import ns as prep_ns
from api.endpoints.trainengine import ns as train_ns

from flask_cors import CORS

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__,instance_relative_config=True)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}});
    
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    print(app.has_static_folder)
    print(app.static_folder)

    app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP
    blueprint = Blueprint('api', __name__, url_prefix='/api/process')
    api = Api(blueprint,version='1.0', title='Flexible Vision Preparation and Training API Gateway',
              description='Do Great things')

    print(api)
    api.add_namespace(prep_ns)
    api.add_namespace(train_ns)
    app.register_blueprint(blueprint)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass



    return app
