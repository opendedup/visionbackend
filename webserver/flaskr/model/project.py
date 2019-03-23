import logging
import settings
from tensorflow_serving.apis import get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import json
import botocore
import boto3
from object_detection.utils import label_map_util
import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format
import shutil
import os
import tempfile
from pathlib import Path
base_dir = './'


global projects

projects = {}


class Project(object):
    def __init__(self, name, parameters):
        self.__dict__ = parameters
        if 'access_key' not in parameters:
            self.access_key = os.environ['ACCESS_KEY']
        if 'secret_key' not in parameters:
            self.secret_key = os.environ['SECRET_KEY']
        if 'S3_URL' not in os.environ:
            self.url = parameters['url']
        else:
            self.url = os.environ['S3_URL']
        if 'bucket' not in parameters:
            self.bucket = os.environ['BUCKET']
        if 'folder' not in parameters:
            self.folder = name
        self.s3 = boto3.client('s3',
                               endpoint_url=self.url,
                               config=boto3.session.Config(
                                   signature_version='s3v4'),
                               aws_access_key_id=self.access_key,
                               aws_secret_access_key=self.secret_key
                               )
        try:
            if not self.keep_temp:
                shutil.rmtree(tempfile.gettempdir() + '/' + name)
        except:
            print('temp dir ' + tempfile.gettempdir() +
                  '/' + name + ' does not exist')
        tmpdir = Path(tempfile.gettempdir() + '/' + name)
        if not tmpdir.is_dir():
            os.mkdir(tempfile.gettempdir() + '/' + name)
        self.tempdir = tempfile.gettempdir() + '/' + name + "/"
        if not hasattr(self, 'predict'):
            self.predict = False
        if self.predict:
            self.label_file = "trained_models/{}/{}/object-detection.pbtxt".format(
                self.model, self.version)
            self.get_classes()
            if 'grpc' not in parameters:
                self.grpc = False
            if self.grpc:
                channel = grpc.insecure_channel(
                    '{}:{}'.format(self.tfserver, self.tfport))
                self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
                    channel)
                request = get_model_metadata_pb2.GetModelMetadataRequest()
                request.model_spec.name = self.model
                request.metadata_field.append("signature_def")
                response = self.stub.GetModelMetadata(request, 10.0)
                print(response.model_spec.version.value)
                print('project [{}] sig [{}]'.format(
                    self.name, response.metadata['detection_signature']))

    def get_classes(self):
        try:
            label_map_string = ''
            print("getting " + self.label_file + ' from ' + self.bucket)
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.label_file)
            label_map_string = obj['Body'].read().decode('utf-8')
            print("got " + self.label_file)
            self.label_map = {}
            self.label_map = labelmap.StringIntLabelMap()
            self.labelmap_dict = {}
            text_format.Merge(label_map_string, self.label_map)
            self.categories = []
            for item in self.label_map.item:
                self.labelmap_dict[item.id] = item.name
                self.categories.append({'id': item.id, 'name': item.name})
            self.category_index = label_map_util.create_category_index(
                self.categories)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                self.labelmap_dict = {}
                print(self.label_file + ' was not found')
            else:
                print(e)
                raise ValueError('Error getting ' +
                                 self.label_file + ' in ' + self.bucket)

    def __getstate__(self):
        state = dict(self.__dict__)
        if 's3' in state:
            del state['s3']
        if 'S3' in state:
            del state['S3']
        if 's3' in state:
            del state['s3']
        if 'label_map' in state:
            del state['label_map']
        if 'labelmap_dict' in state:
            del state['labelmap_dict']
        if 'categories' in state:
            del state['categories']
        if 'category_index' in state:
            del state['category_index']
        if 'stub' in state:
            del state['stub']
        if 'access_key' in state:
            del state['access_key']
        if 'secret_key' in state:
            del state['secret_key']
        if 'url' in state:
            del state['url']
        if 'tempdir' in state:
            del state['tempdir']
        logging.debug(state)
        return state


def clean_state(project):
    state = dict(project)
    if 's3' in state:
        del state['s3']
    if 'S3' in state:
        del state['S3']
    if 's3' in state:
        del state['s3']
    if 'label_map' in state:
        del state['label_map']
    if 'labelmap_dict' in state:
        del state['labelmap_dict']
    if 'categories' in state:
        del state['categories']
    if 'category_index' in state:
        del state['category_index']
    if 'stub' in state:
        del state['stub']
    if 'access_key' in state:
        del state['access_key']
    if 'secret_key' in state:
        del state['secret_key']
    if 'url' in state:
        del state['url']
    

    return state

    

def persist(data):
    pr = {}
    if settings.storage.exists("projects/projects.json"):
        st = settings.storage.download_to_string("projects/projects.json")
        pr = json.loads(st)
    pd = clean_state(data)
    if 'activePhoto' in pd:
        del pd['activePhoto']
    if 'photos' in pd:
        del pd['photos']
    if 'tags' in pd:
        del pd['tags']
    existing_project = data['name'] in pr
    pr[data['name']] = pd

    projects[data['name']] = Project(data['name'], data)
    data['bucket'] = settings.data['BUCKET']
    settings.storage.upload_data(json.dumps(
        pr), 'projects/projects.json', contentType='application/json')
    state_file = 'projects/{}/{}'.format(data['id'], 'state.json')
    if not existing_project and settings.storage.exists(state_file) :
        logging.debug("state file exists for new project {}".format(data['name']))
        pass
    else:
        settings.storage.upload_data(json.dumps(clean_state(
            data)), state_file, contentType='application/json')


def get_projects():
    if settings.storage.exists("projects/projects.json"):
        pr = json.loads(settings.storage.download_to_string(
            "projects/projects.json"))
        ar = []
        for _, v in pr.items():
            v['bucket'] = settings.data['BUCKET']
            ar.append(v)
        return ar
    else:
        return []


def get_project(name):
    pr = {}
    if settings.storage.exists("projects/projects.json"):
        pr = json.loads(settings.storage.download_to_string(
            "projects/projects.json"))

    if name not in pr:
        return None
    pid = pr[name]['id']
    pr = json.loads(settings.storage.download_to_string(
        "projects/{}/state.json".format(pid)))
    pr['bucket'] = settings.data['BUCKET']
    return pr
