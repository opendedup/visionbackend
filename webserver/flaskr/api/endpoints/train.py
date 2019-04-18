from PIL import Image
import io
import base64
import logging

from flask import request
from flask_restplus import Resource, Namespace, fields

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


from api.endpoints.models import category
from model.project import projects
from model.project import get_project
from model.project import persist

import concurrent.futures
import multiprocessing

import logging

import settings
from werkzeug import secure_filename
log = logging.getLogger(__name__)
import utils.tflogs as tflogs

ns = Namespace(
    'train', description='Operations related capturing images from cameras for training')

image = ns.model('Image', {
    'image': fields.String(required=True, description='The name of the image',
                           example="AAAA-SSSS.jpg"),
    'path': fields.String(required=True, description='The path to the image',
                           example="projects/project0/AAAA-SSSS.jpg"),
    'base64': fields.String(required=False, description='The base64 encoded thumbnail of the image',
                           example="ADEW43")
})
#Image for list entry
img_item = ns.model('ImageItem',{
    'name': fields.String(required=True,description='Name of image',example="image.jpg"),
    'ETag': fields.String(required=True,description='ETag returned from storage', 
        example="2206623520-1"),
    'size': fields.Integer(required=True,description='Size of image in bytes',
        example=100038),
    'lastModified': fields.Integer(required=True,
        description='Date image was last modified as a unix time stamp',
        example=1548105112),
    'imgUrl': fields.String(required=True,description='Url to get image',
        example='/api/capture/train/image/project22/image0.jpg')
    
})

snaprequest = ns.model('SnapRequest', {
    'count': fields.Integer(required=True, description='The number of images to snap',
                           example=10),
    'base64': fields.Boolean(required=False, description='If true will return a base64 response',
                           example=False,default=False )
})


eval_summary = ns.model('EvalSummary', {
    'key': fields.String(required=True, description='The number of images to snap',
                           example=10),
    'image_b64': fields.String(required=False, description='A Base64 representation of the JPG Eval Image',
                           example="ADEW43" ),
    'value': fields.Float(required=False, description='The numeric value returned for the key.',
                                example=0.90),
})

train_eval = ns.model('TrainingEval', {
    'timestamp': fields.Integer(required=True, description='Timestamp when eval was completed',
                           example=1553306439506),
    'step': fields.Integer(required=True, description='The training step of the evaluation',
                           example="475697" ),
    'summary':fields.List(fields.Nested(eval_summary,required=True, description='The list of eval statistics and images at this step')),
})

multisnap = ns.model('SnapResponse',{
    'images':fields.List(fields.Nested(image,required=False, description='List of additional images created with snap')),
    'image': fields.String(required=True, description='The name of the image',
                           example="AAAA-SSSS.jpg"),
    'path': fields.String(required=True, description='The path to the image',
                           example="projects/project0/AAAA-SSSS.jpg"),
    'base64': fields.String(required=False, description='The base64 encoded thumbnail of the image',
                           example="ADEW43")
})

training = ns.model('Training',{
    'name': fields.String(required=True,
                            description='The name of the model that will be trained',
                            example='project0'),
    'bucket':fields.String(required=False, description='The s3 bucket target',
                           example='bucket0'),
    'model':fields.String(required=True, description='The pretrained model to start with',
                           example='faster_rcnn_inception_v2_coco',
                           enum=['faster_rcnn_inception_v2_coco',
                                 'ssd_resnet50_v1_fpn_shared_box_predictor_coco14_sync',
                                 'ssd_mobilenet_v1_fpn_shared_box_predictor_coco14_sync']),
    'num_train_steps':fields.Integer(required=True, description='The number of steps to train',
                             example=50000),
    'ml_job_id':fields.String(required=False, description='The job id for the ml engine job',
                           example='imagerie_bdca1eb5_b0d4_422c_83b3_c612cd725c3f'),
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
    'bucket':fields.String(required=True, description='The bucket where the data is located',
                          example='bucket0'),
    'categories':fields.List(fields.Nested(category,required=True, description='The object tags for this training')),
    'training_eval':fields.List(fields.Nested(train_eval,required=True, description='The training evaluation')),
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
    'use_tpu':fields.Boolean(required=False,description='Use GCS for storage.',
                              default=False),
    'project_name': fields.String(required=True, description='The name of the project assigned to this training',
                          example='project0'),
    'end_time' :fields.Integer(required=False, description='The end time of the training as a Unix Timestamp',
                          example=1551641464731)
})

def str2bool(v):
  return v.lower() in ("yes", "true", "1")

def download_jobdef(file,mp):
    fldr = 'training_jobs/'
    mm = None
    try:
        jd = settings.storage.download_to_string(file + 'job_def.json')
        tl = file[len(fldr):]
        t = tl.split('/')
        mm = json.loads(jd)
        mm['name'] = t[0]
        mm['bucket'] = settings.data['BUCKET']
        mp[t[0]]=mm
    except:
        pass
    
    return mm


@ns.route('/cameras/<string:project>/<int:cam>')
@ns.param('project', 'The project name where the image is located')
@ns.param('cam', 'The camera number to capture the image')
class Snap(Resource):
    @ns.response(404, description='Project not found.')
    @ns.response(200, description='return image metadata.')
    @ns.marshal_with(image)
    @jwt_required
    def get(self, cam, project):
        """
        Takes a snapshot and uploads it to the bucket/project_folder.
        """
       
        p = get_project(project)
       
        if p is None:
            return 404
        if not "photos" in p:
            p["photos"] = []
        
        image_name = uuid.uuid4()
        camera = settings.camera.get_cam(cam)
        img, _, _, _ = settings.camera.get_picture(cam)
        ofn = "projects/{}/{}{}".format(p['id'],
                                        image_name, settings.data["image_type"])
        settings.storage.upload_data(img.tobytes(),ofn,contentType='image/jpeg')
        tcapture = {"request": {},"images":[],"base_image":ofn,"base_image_format":settings.data["image_type"],"camera":camera.__getstate__()}
        resp_obj = {"image": str(image_name) +
                                 settings.data["image_type"], "path": ofn}
        image = Image.open(io.BytesIO(img.tobytes()))
        image.thumbnail((128, 128), Image.ANTIALIAS)
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
        purl = "{}/{}{}".format(p['id'],
                                        image_name, settings.data["image_type"])
        height, width = img.shape[:2]

        ptag = {
            "modified": 1548116137598,
            "dimensions": {
                "width": width,
                "height": height,
                "size": len(img.tobytes())
            },
            "photoId": str(image_name),
            "title": "photo:{} - camera:{}".format(len(p["photos"]),cam),
            "expanded": False,
            "imgUrl": purl,
            "children":[]
        }
        p["photos"].append(ptag)
        p["activePhoto"]=ptag
        
        resp_obj['base64'] = ncoded_string
        tcapture["base_image_b64"]=ncoded_string
        tofn = "projects/{}/{}{}".format(p['id'],
                                                image_name, ".tcapture")
        settings.storage.upload_data(json.dumps(tcapture),tofn,contentType='application/json')
        persist(p)
        return resp_obj

    @ns.response(404, description='Project not found.')
    @ns.produces(['text/event-stream'])
    @ns.expect(snaprequest,skip_none=True)
    @jwt_required
    def post(self,cam,project):
        """
        Produces Multiple snapshots from a camera at once responding with an event stream
        """
        data = request.json
        p = get_project(project)
        if p is None:
            return 404
        if not "photos" in p:
            p["photos"] = []
        
        def eventStream(data):
            image_name = uuid.uuid4()
            camera = settings.camera.get_cam(cam)
            img, _, _, _ = settings.camera.get_picture(cam)
            ofn = "projects/{}/{}{}".format(p['id'],
                                            image_name, settings.data["image_type"])
            settings.storage.upload_data(img.tobytes(),ofn,contentType='image/jpeg')
            resp_obj = {"image": str(image_name) +
                                    settings.data["image_type"], "path": ofn}
            tcapture = {"request": data,"images":[],"base_image":ofn,"base_image_format":settings.data["image_type"],"camera":camera.__getstate__()}
            purl = "{}/{}{}".format(p['id'],
                                        image_name, settings.data["image_type"])
            height, width = img.shape[:2]

            ptag = {
                "modified": 1548116137598,
                "dimensions": {
                    "width": width,
                    "height": height,
                    "size": len(img.tobytes())
                },
                "photoId": str(image_name),
                "title": "photo:{} - camera:{}".format(len(p["photos"]),cam),
                "expanded": False,
                "imgUrl": purl,
                "children":[]
            }
            p["photos"].append(ptag)
            p["activePhoto"]=ptag
            persist(p)

            if 'base64' in data and data['base64'] == True:
                image = Image.open(io.BytesIO(img.tobytes()))

                image.thumbnail((128, 128), Image.ANTIALIAS)
                imgByteArr = io.BytesIO()

                image.save(imgByteArr, format='JPEG')
                imgByteArr = imgByteArr.getvalue()
                ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
                resp_obj['base64'] = ncoded_string
                tcapture["base_image_b64"]=ncoded_string
                yield 'data: {}\n\n'.format(resp_obj)
            if 'count' in data:
                for i in range(data['count']-1):
                    img, _, _, _ = settings.camera.get_picture(cam)
                    ofn = "projects/{}/{}/{}{}".format(p['id'],
                                                    image_name,i, settings.data["image_type"])
                    settings.storage.upload_data(img.tobytes(),ofn,contentType='image/jpeg')
                    resp_obj = {"image": str(i)+
                                        settings.data["image_type"], "path": ofn}
                    tcapture["images"].append(ofn)
                    if 'base64' in data and data['base64'] == True:
                        image = Image.open(io.BytesIO(img.tobytes()))

                        image.thumbnail((128, 128), Image.ANTIALIAS)
                        imgByteArr = io.BytesIO()

                        image.save(imgByteArr, format='JPEG')
                        imgByteArr = imgByteArr.getvalue()
                        ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
                        resp_obj['base64'] = ncoded_string
                    yield 'data: {}\n\n'.format(resp_obj)
            tofn = "projects/{}/{}{}".format(p['id'],
                                                image_name, ".tcapture")
            settings.storage.upload_data(json.dumps(tcapture),tofn,contentType='application/json')            
        return Response(eventStream(data), mimetype="text/event-stream")
    
    @ns.response(404, description='Project not found.')
    @ns.marshal_with(multisnap)
    @ns.expect(snaprequest,skip_none=True)
    @jwt_required
    def put(self,cam,project):
        """
        Produces snapshot(s) from a camera selecting additional paramaters
        """
        data = request.json
        p = get_project(project)
        if p is None:
            return 404
        if not "photos" in p:
            p["photos"] = []
        image_name = uuid.uuid4()
        camera = settings.camera.get_cam(cam)
        img, _, _, _ = settings.camera.get_picture(cam)
        ofn = "projects/{}/{}{}".format(p['id'],
                                        image_name, settings.data["image_type"])
        settings.storage.upload_data(img.tobytes(),ofn,contentType='image/jpeg')
        images =[]
        resp_obj = {"image": str(image_name) +
                                 settings.data["image_type"], "path": ofn,'images':images}
        tcapture = {"request": data,"images":images,"base_image":ofn,"base_image_format":settings.data["image_type"],"camera":camera.__getstate__()}
        purl = "{}/{}{}".format(p['id'],
                                        image_name, settings.data["image_type"])
        height, width = img.shape[:2]

        ptag = {
            "modified": 1548116137598,
            "dimensions": {
                "width": width,
                "height": height,
                "size": len(img.tobytes())
            },
            "photoId": str(image_name),
            "title": "photo:{} - camera:{}".format(len(p["photos"]),cam),
            "expanded": False,
            "imgUrl": purl,
            "children":[]
        }
        p["photos"].append(ptag)
        p["activePhoto"]=ptag
        persist(p)

        if 'base64' in data and data['base64'] == True:
            image = Image.open(io.BytesIO(img.tobytes()))

            image.thumbnail((128, 128), Image.ANTIALIAS)
            imgByteArr = io.BytesIO()

            image.save(imgByteArr, format='JPEG')
            imgByteArr = imgByteArr.getvalue()
            ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
            resp_obj['base64'] = ncoded_string
            tcapture["base_image_b64"]=ncoded_string
        if 'count' in data:
            for i in range(data['count']-1):
                img, _, _, _ = settings.camera.get_picture(cam)
                ofn = "projects/{}/{}/{}{}".format(p['id'],
                                                image_name,i, settings.data["image_type"])
                settings.storage.upload_data(img.tobytes(),ofn,contentType='image/jpeg')
                
                images.append({"image": str(i)+
                                    settings.data["image_type"], "path": ofn})
        tofn = "projects/{}/{}{}".format(p['id'],
                                                image_name, ".tcapture")
        settings.storage.upload_data(json.dumps(tcapture),tofn,contentType='application/json')
        return resp_obj



@ns.route('/image/<string:project>/<string:image>')
@ns.param('project', 'The project name where the image is located')
@ns.param('image', 'The relative image filename')
class GetExisting(Resource):
    @ns.produces(['image/jpeg'])
    @ns.response(200, description='return image file.')
    @ns.response(404, description='Image not found.')
    @ns.response(404, description='Project not found.')
    @jwt_required
    def get(self, project, image):
        """
        Returns an existing image in the project.
        """
        p = get_project(project)
        if p is None:
            return 404
        fn = "projects/{}/{}".format(p['id'], image)
        if not settings.storage.exists(fn):
            return 404
        image_data = settings.storage.download_to_data(fn)
        response = make_response(image_data)
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set(
            'Content-Disposition', 'attachment', filename='%s.jpg' % fn)
        return response
    
    @ns.response(204, 'Image successfully deleted.')
    @ns.response(404, 'Image not found.')
    @ns.response(404, description='Project not found.')
    @jwt_required
    def delete(self,project,image):
        """
        Deletes an image from a project.
        """
        p = get_project(project)
        if p is None:
            return "project not found",404
        fn = "projects/{}/{}".format(p['id'],image)
        tfn ="projects/{}/{}".format(p['id'],image.split('.')[0] + '.tcapture')
        purl = "{}/{}".format(project,image)

        if 'activePhoto' in p:
            if p['activePhoto']['imgUrl'] == purl:
                del p['activePhoto']
        photos = []
        if 'photos' in p:
            for photo in p['photos']:
                if photo['imgUrl'] != purl:
                    photos.append(photo)
                    p['activePhoto']=photo
        
        p["photos"] = photos
        persist(p)
        if not settings.storage.exists(fn):
            return 404
        settings.storage.delete_cloud_file(fn)
        if settings.storage.exists(tfn):
            tcapture = json.loads(settings.storage.download_to_string(tfn))
            logging.debug(tcapture)
            for img in tcapture["images"]:
                settings.storage.delete_cloud_file(img['path'])
            settings.storage.delete_cloud_file(tfn)
        return None, 204

@ns.route('/images/<string:project>/')
@ns.param('project', 'The project name where the training images are located')
class GetExistingList(Resource):
    @ns.marshal_list_with(img_item)
    @ns.response(200, description='return a list of images.')
    @ns.response(404, description='Project not found.')
    @jwt_required
    def get(self, project):
        """
        Returns a list of all images available for training.
        """
        p = get_project(project)
        if p is None:
            return 404
        fn = "projects/{}/".format(p['id'])
        lst = settings.storage.list_files(fn,delimiter='/',metadata=True)
        rl = []
        for t in lst:
            print(t)
            if t['name'].endswith('.jpg'):
                t['name'] = t['name'][len(fn):]
                t['imgUrl'] = project + '/' + t['name']
                rl.append(t)
        return rl
       
       

@ns.route('/cameras/stream/<int:cam>')
@ns.param('cam','The camera number to capture the stream')
class Stream(Resource):
    @ns.produces(['multipart/x-mixed-replace; boundary=--jpgboundary'])
    @jwt_required
    def get(self,cam):
        """
        Streams the video from the designated camera.
        """
        def gen(cam):
            """Video streaming generator function."""
            while True:
                frame = settings.camera.getstream(cam)
                yield (b'--jpgboundary\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
        return Response(gen(cam),mimetype='multipart/x-mixed-replace; boundary=--jpgboundary')

@ns.route('/traingjobs/')
class TrainingJobCollection(Resource):
    @ns.marshal_list_with(training)
    @jwt_required
    def get(self):
        """
        Returns list of Traing Jobs.
        """
        fldr = 'training_jobs/'
        lst = settings.storage.list_files(fldr,delimiter='/',folders_only=True)
        mp = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
            future_img = {executor.submit(
                download_jobdef, l, mp): l for l in lst}
            for future in concurrent.futures.as_completed(future_img):
                #img = future_img[future]
                try:
                    future.result()
                except Exception as exc:
                    var = traceback.format_exc()
                    log.error(var)    
        return list(mp.values())

@ns.route('/eval/<string:name>')
class TrainingEval(Resource):
    @ns.marshal_list_with(train_eval)
    @jwt_required
    def get(self,name):
        """
        Returns latest training evaluation.
        """
        fldr = 'training_jobs/' + name    
        return tflogs.get_events(fldr,'/tmp',settings.storage,full=True,dim=720)


@ns.route('/traingjobs/<string:name>')
@ns.response(404, 'Training Job not found.')
class TrainingJobItem(Resource):

    @ns.marshal_with(training)
    @jwt_required
    def get(self, name):
        """
        Returns a training job.
        """
        fl = 'training_jobs/' + name + '/job_def.json'
        if not settings.storage.exists(fl):
            return 'Corpus not found',404
        else:
            jd = settings.storage.download_to_string(fl)
            mm = json.loads(jd)
            mm['name'] = name
            mm['bucket'] = settings.data['BUCKET']
            return mm
    @ns.response(204, 'Model successfully deleted.')
    @jwt_required
    def delete(self, name):
        """
        Deletes a training job.
        """
        fl = 'training_jobs/' + name + '/job_def.json'
        if not settings.storage.exists(fl):
            return 'Training Job not found',404
        else:
            settings.storage.delete_cloud_dir("training_jobs/{}/".format(name))
        return None, 204

