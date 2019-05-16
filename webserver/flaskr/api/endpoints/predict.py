import logging

from flask import request
from flask_restplus import Resource,Namespace

import boto3
import numpy as np
import uuid
import json

from pathlib import Path

import datetime
import os

from flask import jsonify
from flask import make_response
from flask import g
from flask import request
from flask import Response

from flask_jwt_extended import jwt_required, get_jwt_identity

from model.cameras import cameras
from model.project import Project
from model.project import projects
from model.project import get_project
import settings

from werkzeug import secure_filename

from PIL import Image
import io
import base64


log = logging.getLogger(__name__)



ns = Namespace('predict', description='Operations related capturing images from cameras for prediction')





@ns.route('/snap/<string:project>/<int:cam>')
class Snap(Resource):
    def get(self,cam,project):
        """
        Takes a snapshot and uploads it to the bucket/project_folder/predictions/ and
        returns a json document with the prediction info.
        * coords = relative coordinates of the capture at a percentage of image
        * image = Image name
        * path = Path to the image in the bucket
        * tags = Predition probability for each tag
        ```
        {
            "coords":{
                "c2_good":[0.209846, 0.0299885, 1, 0.637807],
                "emmc_good":[0.344515, 0.593382, 0.500586, 0.694372],
                "hdmi_good":[0.315496, 0.896179, 0.623717, 0.960438],
                "rpi3_cpu_good":[0.155855, 0.417758, 0.295672, 0.517903]
            },
            "image": "92b7245f-970a-4ec5-8fd5-4377b809ec07.jpg",
            "path": "project0/predictions/92b7245f-970a-4ec5-8fd5-4377b809ec07.jpg",
            "tags":{
                "c2_good": 0.422927,
                "emmc_good": 0.640976,
                "hdmi_good": 0.432281,
                "num_detections": "9",
                "rpi3_cpu_good": 0.535758
            }
        }
        ```
        """
        if not project in projects:
            pr = get_project(project)
            projects[project] = Project(project,pr)
            if pr is None:
                return 'Project not found.',404
        p = projects[project]
        image_name = uuid.uuid4()
        img,tags,detections,ppmm = settings.camera.get_picture(cam,predict=True,project=p)
        ofn = "projects/{}/predictions/images/{}{}".format(p.id,image_name,settings.data["image_type"])
        dfn = "projects/{}/predictions/data/{}{}".format(p.id,image_name,settings.data["image_type"])
        p.s3.put_object(Body=img.tobytes(), Bucket=p.bucket, Key=ofn,ContentType='image/jpeg')
        resp_obj= {"image":str(image_name) +settings.data["image_type"],"path":ofn,'tags':tags,'detections' : detections,'ppmm':ppmm,
            'model':p.model, 'model_version':p.version}
        p.s3.put_object(Body=json.dumps(resp_obj), Bucket=p.bucket, Key=dfn,ContentType='application/json')
        return jsonify(resp_obj)

@ns.route('/upload/<string:project>')
class Upload(Resource):
    @jwt_required
    def post(self,project):
        """
        Predicts results based on uploaded file.
        """
        print('uploading')
        if not project in projects:
            pr = get_project(project)
            projects[project] = Project(project,pr)
            if pr is None:
                return 'Project not found.',404
        p = projects[project]
        f = request.files['images']
        p = projects[project]
        fl = request.files['images']
        image_name = fl.filename
        fn = secure_filename(p.tempdir + image_name)
        try:
            fl.save(fn)
            im=Image.open(fn)
            if not im.format == 'JPEG':
                im = im.convert("RGB")
                im.save(fn + '.jpg')
                os.remove(fn)
                fn = fn + '.jpg'
            
            image_name = str(uuid.uuid4()) + '.jpg'
            img,bimg,tags,detections,ppmm = settings.camera.read_picture(fn,predict=True,project=p)
            with open(fn, 'rb') as file_t:
                blob_data = bytearray(file_t.read())
                ofn = "projects/{}/predictions/images/{}{}".format(p.id,image_name,settings.data["image_type"])
                settings.storage.upload_data(blob_data,ofn,contentType='image/jpeg')
            ofn = "{}/predictions/data/{}.{}".format(project,image_name,'json')
            image = Image.open(io.BytesIO(bimg.tobytes()))
            image.thumbnail((640, 640), Image.ANTIALIAS)
            imgByteArr = io.BytesIO()
            image.save(imgByteArr, format='JPEG')
            imgByteArr = imgByteArr.getvalue()
            ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
            resp_obj= {"image":str(image_name) +settings.data["image_type"],"path":ofn,'tags':tags,'detections' : detections,'ppmm':ppmm,
            'model':p.model, 'model_version':p.version,'base64':ncoded_string}
            settings.storage.upload_data(json.dumps(resp_obj), ofn ,contentType='application/json')
        finally:
            os.remove(fn)
        return jsonify(resp_obj)

@ns.route('/snap/<string:project>/<string:image>')
class GetExisting(Resource):
    @ns.produces(['multipart/x-mixed-replace; boundary=--jpgboundary'])
    @jwt_required
    def get(self,project,image):
        """
        Returns an existing image in the project.
        """
        if not project in projects:
            pr = get_project(project)
            projects[project] = Project(project,pr)
            if pr is None:
                return 'Project not found.',404
        p = projects[project]
        fn = "projects/{}/predictions/images/{}".format(p.id,image)
        obj = p.s3.get_object(Bucket=data["bucket"], Key=fn)
        image_data = obj['Body'].read()
        response = make_response(image_data)
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set(
            'Content-Disposition', 'attachment', filename='%s.jpg' % fn)
        return response

@ns.route('/stream/<string:project>/<int:cam>')
class Stream(Resource):

    #@settings.api.representation('multipart/x-mixed-replace; boundary=--jpgboundary')
    @ns.produces(['multipart/x-mixed-replace; boundary=--jpgboundary'])
    @jwt_required
    def get(self,project,cam):
        """
        Streams the video from the designated camera.
        """
        if not project in projects:
            pr = get_project(project)
            projects[project] = Project(project,pr)
            if pr is None:
                return 'Project not found.',404
        p = projects[project]
        def gen(cam,project):
            """Video streaming generator function."""
            while True:
                frame = settings.camera.getstream(cam,predict=True,project=project)
                yield (b'--jpgboundary\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
        return Response(gen(cam,p),mimetype='multipart/x-mixed-replace; boundary=--jpgboundary')
