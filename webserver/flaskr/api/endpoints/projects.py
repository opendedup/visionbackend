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

import settings

from flask import jsonify
from flask import make_response
from flask import g
from flask import request
from flask import Response

from flask_jwt_extended import jwt_required, get_jwt_identity

from model.cameras import Cameras
from model.project import Project
from model.project import get_project
from model.project import get_projects
from model.project import projects
from model.project import persist

log = logging.getLogger(__name__)
ux_exampl={"cat1":{"name":"Adhesives","color":"#5fc27e","count":95,"alpha":50},
    "cat2":{"name":"Screws","color":"#F23B4B","count":0,"alpha":50}}
camera_exampl="{}"
tags_exampl={"Pins":{
"name":"Pins",
"count":22,
"desc":""
},
"RPI3":{
"name":"RPI3",
"count":5,
"desc":""
},
"HDMI":{
"name":"HDMI",
"count":13,
"desc":""
},
"RPI3_CPU":{
"name":"RPI3_CPU",
"count":5,
"desc":""
},
"RPI3_DISPLAY":{
"name":"RPI3_DISPLAY",
"count":5,
"desc":""
},
"RPI3_CAMERA":{
"name":"RPI3_CAMERA",
"count":5,
"desc":""
}}

ns = Namespace('project', description='Manage Vision Projects')

project = ns.model('Project', {
    'name': fields.String(required=True, description='The project name',
                          example='project0'),
    'id': fields.String(required=True, description='The s3 bucket subfolder',
                        example='zaabc'),
    'predict': fields.Boolean(required=True, description='Boolean to detemine ' +
                              ' if this project is predictive',
                              default=False),
    'model': fields.String(required=False, description='The model used for ' +
                           ' prediction',
                           example='model0'),
    'version': fields.String(required=False, description='The model version used for ' +
                             ' prediction',
                             example="147891234"),
    'mlengine': fields.Boolean(required=False, description='If true use mlengine ' +
                               'for prediction',
                               default=False),
    'tfserver': fields.String(required=False, description='The IP or host name ' +
                              'of the tensorflow serving server.',
                              example='tfserver0'),
    'tfport': fields.Integer(required=False, description='The port of the tensorflow ' +
                             'serving server',
                             example=8500),
    'sensitivity': fields.Float(required=False, description='Probability Sensitivity for Ojbect Detection.',
                                example=0.90),
    'created': fields.Integer(required=False, description='Timestamp of Date Created.',
                              example=1544978140522),
    'modified': fields.Integer(required=False, description='Timestamp of Date modified.',
                               example=1545353804201),
    'ux': fields.Raw(required=False,example=ux_exampl),
    'cameras': fields.Raw(required=False),
    'tags': fields.Raw(required=False, description="map of object tags"),
    'activePhoto': fields.Raw(required=False, description="currently active photo"),
    'photos':fields.List(fields.Raw()),
    
    'notificationEmail': fields.String(required=False, description='Not Needed',
                              example='email@domain.com'),
    'bucket' : fields.String(required=False, description='The bucket where the data is located',
                          example='bucket0'),
    'searchString': fields.String(),
    'searchFocusIndex': fields.Integer(),
    'searchFoundCount': fields.Integer(),
    
    'nodeUrl': fields.String(),

})


listed_projects = ns.model('ListedProjects', {
    'name': fields.String(required=True, description='The Project name'),
    'project': fields.Nested(project, description='The Project')
})

@ns.route('/')
class ProjectCollection(Resource):
    @ns.marshal_list_with(project)
    @jwt_required
    def get(self):
        """
        Returns list of Projects.
        """
        return get_projects()

    @ns.response(201, 'Project successfully created.')
    @ns.expect(project,skip_none=True)
    @jwt_required
    def post(self):
        """
        Creates a new project.
        Use this method to create a new project.
        Projects are used to determine path for bucket location and
        prediction engine path.
        """
        data = request.json
        if settings.storage.exists("projects/projects.json"):
            pr = json.loads(settings.storage.download_to_string("projects/projects.json"))
            if data['name'] in projects:
                return 'object exists with that name',409
        logging.debug(data)
        
        persist(data)
        return 'Project successfully created.', 201

@ns.route('/<string:name>')
@ns.response(404, 'Project not found.')
class ProjectItem(Resource):

    @ns.marshal_with(project)
    @jwt_required
    def get(self, name):
        """
        Returns a project.
        """
        pr = get_project(name);
        if pr is None:
             return 'Project not found.',404
        return pr

    @ns.expect(project,skip_none=True)
    @ns.response(204, 'Category successfully updated.')
    @jwt_required
    def put(self, name):
        """
        Updates a Project.
        Use this method to change attributes of a project.
        * Send a JSON object to turn on prediction for a project.
        ```
        {
          "predict":true
        }
        ```
        * Specify the name of the category to modify in the request URL path.
        """
        data = request.json
        if settings.storage.exists("projects/projects.json"):
            pr = json.loads(settings.storage.download_to_string("projects/projects.json"))
        else:
            return {'error':'no projects exist'},404
        if name not in pr:
            return {'error':'no project exists with that name'},404
        if name in data:
            return {'error':'project name should not be included in json request'},422
        
        pid = pr[name]['id']
        opr = json.loads(settings.storage.download_to_string("projects/{}/state.json".format(pid)))
        
        z = {**opr, **data}
        persist(z)
        return None, 204

    @ns.response(204, 'Project successfully deleted.')
    @jwt_required
    def delete(self, name):
        """
        Deletes a project.
        """
        if settings.storage.exists("projects/projects.json"):
            pr = json.loads(settings.storage.download_to_string("projects/projects.json"))
        else:
            return 'no project exists with that name',404
        if name not in pr:
            return {'error':'no project exists with that name'},404
        del pr[name]
        settings.storage.upload_data(json.dumps(pr),"projects/projects.json",contentType='application/json')
        return None, 204
