# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from googleapiclient.errors import HttpError
from googleapiclient import errors
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import traceback
import json
import yaml
from os.path import dirname
import pathlib
import time
from datetime import datetime
from rq import get_current_job
import uuid
import utils.exporter as exporter
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.utils.config_util import create_pipeline_proto_from_configs
import fileinput
from models.job import Job
from pathlib import Path
import subprocess
import sys
import os
import shutil
import io
from object_detection import model_lib
from object_detection import model_hparams




from absl import flags

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


base_dir = './'


eval_training_data = False
sample_1_of_n_eval_examples = 1
sample_1sample_1_of_n_eval_examples = 5
hparams_overrides = None
run_once = False
checkpoint_dir = None

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def updateFile(job):
    shutil.copytree('tfpipeline/tmodels/'+job.model +
                    '/', job.tempdir + '/' + job.model)
    shutil.copyfile('tfpipeline/tmodels/'+job.model+'.config', job.tempdir +
                    '/' + job.model + '/pipeline.config')
    fl = job.tempdir + '/' + job.model + '/pipeline.config'
    job.pipeline = fl
    tmpdir = Path(job.tempdir + '/corpus')
    if not tmpdir.is_dir():
        os.mkdir(job.tempdir + '/corpus')
    with open(fl, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('@NUM_CLASS', str(len(job.categories)))
    filedata = filedata.replace('@MAX_DIM', str(job.max_dim))
    filedata = filedata.replace('@MIN_DIM', str(job.min_dim))
    filedata = filedata.replace(
        '@CHK_PATH', job.tempdir + '/' + job.model + '/model.ckpt')
    filedata = filedata.replace(
        '@LABEL_PATH', job.tempdir + '/corpus/object-detection.pbtxt')
    filedata = filedata.replace(
        '@TRAIN_PATH', job.tempdir + '/corpus/train-record-?????-of-00010')
    filedata = filedata.replace(
        '@TEST_PATH', job.tempdir + '/corpus/validation-record-?????-of-00010')
    filedata = filedata.replace('@BATCH_SIZE', str(job.batch_size))
    filedata = filedata.replace('@NUM_STEPS', str(job.num_train_steps))
    filedata = filedata.replace('@NUM_EVALS', job.num_evals)
    with open(fl, 'w') as file:
        file.write(filedata)


def updateFileML(job):
    fl = '/svbackend/tfpipeline/tmodels/'+job.model+'.config'
    with open(fl, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('@NUM_CLASS', str(len(job.categories)))
    filedata = filedata.replace('@MAX_DIM', str(job.max_dim))
    filedata = filedata.replace('@MIN_DIM', str(job.min_dim))
    prep_model = 'gs://'+job.bucket+'/pretrained/'+job.model+'/model.ckpt'
    filedata = filedata.replace('@CHK_PATH', prep_model)
    labels = 'gs://'+job.bucket+'/corpus/' + \
        job.prep_name + "/object-detection.pbtxt"
    filedata = filedata.replace('@LABEL_PATH', labels)
    train_records = 'gs://'+job.bucket+'/corpus/' + \
        job.prep_name + "/train-record-?????-of-00010"
    filedata = filedata.replace('@TRAIN_PATH', train_records)
    test_records = 'gs://'+job.bucket+'/corpus/' + \
        job.prep_name + "/validation-record-?????-of-00010"
    filedata = filedata.replace('@TEST_PATH', test_records)
    filedata = filedata.replace('@BATCH_SIZE', str(job.batch_size))
    filedata = filedata.replace('@NUM_STEPS', str(job.num_train_steps))
    filedata = filedata.replace(
        '@EVAL_PATH', 'gs://'+job.bucket+'/training_jobs/' + job.name+'/evals')
    filedata = filedata.replace('@NUM_EVALS', str(job.num_evals))
    pipeling_key = 'training_jobs/' + job.name+'/running_pipeline.config'
    job.upload_data(filedata, pipeling_key, contentType='text/plain')
    job.copy_object("corpus/" + job.prep_name + "/object-detection.pbtxt",
                    'training_jobs/' + job.name+'/object-detection.pbtxt')
    job.pipeline = 'gs://'+job.bucket+'/'+pipeling_key

    if hasattr(job, 'use_tpu') and job.use_tpu:
        logger.debug('using tpu')


def upload_model(job):
    key = 'pretrained/'+job.model + '/'
    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(prefix=key)
        for blob in blobs:
            logger.error("model exists in bucket " +
                         key + ' blob=' + blob.name)
            return

    else:
        resp = job.s3.list_objects(Bucket=job.bucket, Prefix=key)
        if 'Contents' in resp and len(resp['Contents']) > 0:
            logger.error("model exists in bucket " + key)
            return
    local_directory = '/svbackend/tfpipeline/tmodels/'+job.model+'/'
    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):

        for filename in files:

            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)

            s3_path = os.path.join("pretrained/" + job.model, relative_path)

            # relative_path = os.path.relpath(os.path.join(root, filename))

            logger.debug("Uploading {}...".format(s3_path))
            job.upload_file(local_path, s3_path)


def upload_packages(job):
    key = 'packages/'
    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(prefix=key)
        for blob in blobs:
            logger.debug("packages exists in bucket " + key)
            return

    else:
        resp = job.s3.list_objects(Bucket=job.bucket, Prefix=key)
        if 'Contents' in resp and len(resp['Contents']) > 0:
            logger.debug("packages exists in bucket " + key)
            return
    local_directory = '/svbackend/mlpackages/'
    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):

        for filename in files:

            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)

            s3_path = os.path.join("packages/", relative_path)

            logger.debug("Uploading {} to {}...".format(local_path, s3_path))
            job.upload_file(local_path, s3_path)
            logger.debug("Done Uploading {} to {}...".format(
                local_path, s3_path))


def start_ml_engine(job):
    ml_job_path = "gs://" + job.bucket + "/training_jobs/" + job.name + "/"
    mlpackages = ["gs://" + job.bucket + "/packages/object_detection-0.1.tar.gz",
                  "gs://" + job.bucket + "/packages/slim-0.1.tar.gz",
                  "gs://" + job.bucket + "/packages/pycocotools-2.0.tar.gz"]
    job.ml_job_path = ml_job_path
    if hasattr(job, 'jb'):
        job.ml_job_id = "imagerie_"+job.jb.id.replace("-", "_")
    else:
        job.ml_job_id = "imagerie_" + str(uuid.uuid4()).replace("-", "_")
    cmd = ['--model_dir', ml_job_path,
           '--pipeline_config_path', job.pipeline]
    if not hasattr(job, "mle_region"):
        job.mle_region = 'us-central1'
    if hasattr(job, 'use_tpu') and job.use_tpu:
        training_inputs = {'scaleTier': 'CUSTOM',
                           'masterType': 'n1-highmem-8',
                           'workerType': 'cloud_tpu',
                           'workerConfig': {'acceleratorConfig': {'type': 'TPU_V2', 'count': 8}},
                           'workerCount': 1,
                           'packageUris': mlpackages,
                           'pythonModule': 'object_detection.model_tpu_main',
                           'args': cmd,
                           'region': job.mle_region,
                           'jobDir': ml_job_path,
                           'runtimeVersion': '1.12',
                           'pythonVersion': '3.5'}

        cmd.append(['--tpu_zone', job.mle_region])
    else:
        if not hasattr(job, "parameter_servers") or job.parameter_servers <= 0:
            job.parameter_servers = int(job.ml_workers)

        if job.ml_workers <= 1:
            training_inputs = {'scaleTier': 'CUSTOM',
                               'masterType': 'n1-highmem-32',
                               'masterConfig': {'acceleratorConfig': {'count': 4, 'type': 'NVIDIA_TESLA_P100'}},
                               'packageUris': mlpackages,
                               'pythonModule': 'object_detection.model_main',
                               'args': cmd,
                               'region': job.mle_region,
                               'jobDir': ml_job_path,
                               'runtimeVersion': '1.12',
                               'pythonVersion': '3.5'}
        else:
            training_inputs = {'scaleTier': 'CUSTOM',
                               'masterType': 'complex_model_l_gpu',
                               'workerType': 'standard_p100',
                               'parameterServerType': 'standard',
                               'workerCount': job.ml_workers,
                               'parameterServerCount': job.parameter_servers,
                               'packageUris': mlpackages,
                               'pythonModule': 'object_detection.model_main',
                               'args': cmd,
                               'region': job.mle_region,
                               'jobDir': ml_job_path,
                               'runtimeVersion': '1.12',
                               'pythonVersion': '3.5'}
    job_spec = {'jobId': job.ml_job_id, 'trainingInput': training_inputs}
    project_id = 'projects/{}'.format('$GCP_PROJECT')
    cloudml = discovery.build('ml', 'v1')
    request = cloudml.projects().jobs().create(body=job_spec,
                                               parent=project_id)
    try:
        response = request.execute()
        logging.info(response)
        if hasattr(job, 'jb'):
            job.jb.ml_exec = job_spec
            job.jb.exec_output = response
            job.jb.exec_error = None
            job.jb.save_meta()
        while True:
            jobId = '{}/jobs/{}'.format(project_id, job.ml_job_id)
            request = cloudml.projects().jobs().get(name=jobId)
            response = request.execute()
            logger.info(response)
            if hasattr(job, 'jb'):
                job.jb.ml_exec = job_spec
                job.jb.ml_status = response
                job.jb.save_meta()
            time.sleep(30)
            if response['state'] == 'SUCCEEDED' or response['state'] == 'FAILED' or response['state'] == 'CANCELLED':
                return response['state']

        # You can put your code for handling success (if any) here.
    except errors.HttpError as err:
        # Do whatever error response is appropriate for your application.
        # For this example, just send some text to the logs.
        # You need to import logging for this to work.
        logging.error('There was an error creating the training job.'
                      ' Check the details:')
        logging.error(err._get_reason())
        if hasattr(job, 'jb'):
            job.jb.ml_exec = job_spec
            job.jb.exec_output = None
            job.jb.exec_error = err._get_reason()
            job.jb.save_meta()


def create_model_pbtxt(job):
    out_file = 'trained_models/model.config'
    flr_lst = []
    if job.use_gcs:
        # add http=whatever param if auth
        sclient = discovery.build('storage', 'v1')
        request = sclient.objects().list(
            bucket=job.bucket,
            prefix="trained_models/",
            delimiter="/")
        response = request.execute()
        for s in response['prefixes']:
            flr_lst.append(s)
    else:
        while True:
            kwargs = {'Bucket': job.bucket,
                      'Prefix': 'trained_models/',
                      'Delimiter': '/'}
            resp = job.s3.list_objects_v2(**kwargs)
            try:
                for obj in resp['CommonPrefixes']:
                    flr_lst.append(obj['Prefix'][len(kwargs['Prefix']):])
            except KeyError:
                break
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break
    ptxt = 'model_config_list {\n'
    for s in flr_lst:
        s = s[len('trained_models/'):]
        ptxt += "\tconfig {\n"
        ptxt += "\t\tname: '{}'\n".format(s[:len(s)-1])
        ptxt += "\t\tbase_path: 's3://{}/trained_models/{}'\n".format(
            job.bucket, s)
        ptxt += "\t\tmodel_platform: \'tensorflow\'\n"
        ptxt += "\t\tmodel_version_policy: {all: {}}\n"
        ptxt += '\t}\n'
    ptxt += "}"
    job.upload_data(ptxt, out_file, contentType='text/plain')


def deploy_ml_engine(job):
    ml_version_path = "gs://" + job.bucket + "/trained_models/" + \
        job.name + "/" + str(job.model_version)+"/"
    ml_exec = ('/usr/local/gcloud/google-cloud-sdk/bin/gcloud ml-engine models create {} --enable-logging '
               ' --labels=job_id={}'
               ).format(job.name, job.jb.id.replace("-", "_"))
    logger.debug(ml_exec)
    process = subprocess.Popen(ml_exec.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if hasattr(job, 'jb'):
        job.jb.model_exec_output = output
        job.jb.model_exec_error = error
        job.jb.save_meta()
    ml_exec = ('/usr/local/gcloud/google-cloud-sdk/bin/gcloud ml-engine versions create v{} --model={} '
               '--runtime-version=1.10 --python-version=2.7 --framework=tensorflow --labels=job_id={} --origin={}'
               ).format(job.model_version, job.name, job.jb.id.replace("-", "_"), ml_version_path)
    logger.debug(ml_exec)
    process = subprocess.Popen(ml_exec.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if hasattr(job, 'jb'):
        job.jb.version_exec_output = output
        job.jb.version_exec_error = error
        job.jb.save_meta()


def train(job):
    config = tf.estimator.RunConfig(job.tempdir + '/model')
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(hparams_overrides),
        pipeline_config_path=job.pipeline,
        train_steps=job.num_train_steps,
        sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            sample_1sample_1_of_n_eval_examples))
    logger.debug(train_and_eval_dict)
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

def download_eval(job):
    modeldir = job.tempdir + '/eval_0'
    os.mkdir(modeldir)
    tp = 'training_jobs/'+job.name + '/eval_0'
    kwargs = {'Bucket': job.bucket,
              'Prefix': 'training_jobs/'+job.training+ '/eval_0'} 
    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(prefix='training_jobs/'+job.training + '/eval_0')
        for blob in blobs:
            _key = blob.name
            logger.debug("downloading " + _key + " [" + _key[len(tp):] + "]")
            if not _key[len(tp):] == '/':
                try:
                    if _key.endswith('/'):
                        pathlib.Path(
                            modeldir + _key[len(tp):]).mkdir(parents=True, exist_ok=True)
                    else:
                        blob.download_to_filename(modeldir + _key[len(tp):])
                except (FileNotFoundError, NotADirectoryError):
                    parent = dirname(modeldir + _key[len(tp):])
                    pathlib.Path(parent).mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(modeldir + _key[len(tp):])
    else:
        while True:
            resp = job.s3.list_objects_v2(**kwargs)
            try:
                for obj in resp['Contents']:
                    _key = obj['Key']
                    logger.debug("downloading " + _key +
                                 " [" + _key[len(tp):] + "]")
                    if not _key[len(tp):] == '/':
                        try:
                            if _key.endswith('/'):
                                pathlib.Path(
                                    modeldir + _key[len(tp):]).mkdir(parents=True, exist_ok=True)
                            else:
                                job.s3.download_file(
                                    job.bucket, _key, modeldir + _key[len(tp):])
                        except (FileNotFoundError, NotADirectoryError):
                            parent = dirname(modeldir + _key[len(tp):])
                            pathlib.Path(parent).mkdir(
                                parents=True, exist_ok=True)
                            job.s3.download_file(
                                job.bucket, _key, modeldir + _key[len(tp):])
            except KeyError:
                break
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break
    eval_dir = Path(modeldir)
    if eval_dir.is_dir():
        fls = os.listdir(modeldir)
        logger.debug (fls)
        if len(fls) > 0:
            get_eval(job,modeldir + '/' +fls[0])

def download_training(job):
    modeldir = job.tempdir + '/model'
    os.mkdir(modeldir)
    tp = 'training_jobs/'+job.training
    kwargs = {'Bucket': job.bucket,
              'Prefix': 'training_jobs/'+job.training}
    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(prefix='training_jobs/'+job.training)
        for blob in blobs:
            _key = blob.name
            logger.debug("downloading " + _key + " [" + _key[len(tp):] + "]")
            if not _key[len(tp):] == '/':
                try:
                    if _key.endswith('/'):
                        pathlib.Path(
                            modeldir + _key[len(tp):]).mkdir(parents=True, exist_ok=True)
                    else:
                        blob.download_to_filename(modeldir + _key[len(tp):])
                except (FileNotFoundError, NotADirectoryError):
                    parent = dirname(modeldir + _key[len(tp):])
                    pathlib.Path(parent).mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(modeldir + _key[len(tp):])
    else:
        while True:
            resp = job.s3.list_objects_v2(**kwargs)
            try:
                for obj in resp['Contents']:
                    _key = obj['Key']
                    logger.debug("downloading " + _key +
                                 " [" + _key[len(tp):] + "]")
                    if not _key[len(tp):] == '/':
                        try:
                            if _key.endswith('/'):
                                pathlib.Path(
                                    modeldir + _key[len(tp):]).mkdir(parents=True, exist_ok=True)
                            else:
                                job.s3.download_file(
                                    job.bucket, _key, modeldir + _key[len(tp):])
                        except (FileNotFoundError, NotADirectoryError):
                            parent = dirname(modeldir + _key[len(tp):])
                            pathlib.Path(parent).mkdir(
                                parents=True, exist_ok=True)
                            job.s3.download_file(
                                job.bucket, _key, modeldir + _key[len(tp):])
            except KeyError:
                break
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break


def download_training_data(job):
    corpusdir = job.tempdir + '/corpus/'
    os.mkdir(corpusdir)
    tp = 'corpus/'+job.prep_name
    kwargs = {'Bucket': job.bucket,
              'Prefix': 'corpus/'+job.prep_name}
    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(prefix='corpus/'+job.prep_name)
        for blob in blobs:
            _key = blob.name
            logger.debug(_key + " [" + _key[len(tp):] + "]")
            if not _key[len(tp):] == '/':
                try:
                    blob.download_to_filename(corpusdir + _key[len(tp):])
                except FileNotFoundError:
                    parent = dirname(corpusdir + _key[len(tp):])
                    pathlib.Path(parent).mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(corpusdir + _key[len(tp):])
    else:
        while True:
            resp = job.s3.list_objects_v2(**kwargs)
            try:
                for obj in resp['Contents']:
                    _key = obj['Key']
                    logger.debug(_key + " [" + _key[len(tp):] + "]")
                    if not _key[len(tp):] == '/':
                        try:
                            job.s3.download_file(
                                job.bucket, _key, corpusdir + _key[len(tp):])
                        except FileNotFoundError:
                            parent = dirname(corpusdir + _key[len(tp):])
                            pathlib.Path(parent).mkdir(
                                parents=True, exist_ok=True)
                            job.s3.download_file(
                                job.bucket, _key, corpusdir + _key[len(tp):])
            except KeyError:
                break
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break


def export(job, input_type='image_tensor', serving="True"):
    config_pathname = job.tempdir + '/model/running_pipeline.config'

    # Input checkpoint for the model to be exported
    # Path to the directory which consists of the saved model on disk (see above)
    trained_model_dir = job.tempdir + '/model/'

    # Create proto from model confguration
    configs = get_configs_from_pipeline_file(config_pathname)
    pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

    # Read .ckpt and .meta files from model directory
    checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    if not hasattr(job, 'model_version'):
        # Model Version
        dt = datetime.now()
        model_version_id = int(dt.timestamp()*1000)
        job.model_version = model_version_id
    else:
        model_version_id = job.model_version
    # Output Directory
    output_directory = job.tempdir + '/trained/' + str(model_version_id)
    exporter.export_inference_graph(input_type=input_type, pipeline_config=pipeline_proto,
                                    trained_checkpoint_prefix=input_checkpoint, output_directory=output_directory)
    eval_dir = Path(trained_model_dir +'/eval_0')
    if eval_dir.is_dir():
        fls = os.listdir(eval_dir)
        if len(fls) > 0:
            get_eval(job,eval_dir + '/' +fls[0])
            

def get_eval(job,fl):
    events = []

    for e in tf.train.summary_iterator(fl):
        if hasattr(e,'summary'):
            evt = {}
            evt['timestamp'] = int(e.wall_time*1000)
            evt['step'] = e.step
            evt['summary'] = []
            for v in e.summary.value:
                if hasattr(v,'simple_value'):
                    value = {}
                    value['key'] = v.tag
                    value['value'] = v.simple_value
                    evt['summary'].append(value)
            if len(evt['summary']) > 0:
                events.append(evt)
    job.evaluation = events

def upload_metadata(job, dir, history):
    dt = datetime.now()
    job.end_time = int(dt.timestamp()*1000)
    history['history'].append(job.to_export())
    job.upload_data(json.dumps(history), "{}/{}".format(dir,
                                                        "job_history.json"), contentType='application/json')
    job.upload_data(job.to_json_string(), "{}/{}".format(dir,
                                                         "job_def.json"), contentType='application/json')


def upload_labels(job, dir):
    job.upload_data(json.dumps(job.labelmap_dict), "{}/{}".format(dir,
                                                                  "lables.json"), contentType='application/json')


def upload_training(job):
    local_directory = job.tempdir + '/trained/'
    objfile = job.tempdir + '/model/object-detection.pbtxt'
    job.upload_file(objfile, "trained_models/" + job.name +
                    "/"+str(job.model_version)+"/object-detection.pbtxt")
    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)
            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)

            s3_path = os.path.join("trained_models/" + job.name, relative_path)
            job.model_path = "trained_models/" + job.name
            # relative_path = os.path.relpath(os.path.join(root, filename))

            try:
                client.head_object(Bucket=bucket, Key=s3_path)
                logger.debug("Path found on S3! Skipping {}".format(s3_path))

                # try:
                # client.delete_object(Bucket=bucket, Key=s3_path)
                # except:
                # print "Unable to delete %s..." % s3_path
            except:
                logger.debug("Uploading {}...".format(s3_path))
                job.upload_file(local_path, s3_path)


def train_job(job):
    jb = get_current_job()
    dt = datetime.now()
    jb.meta['job_exec_time'] = str(int(dt.timestamp()*1000))
    logger.debug('Current job: %s' % (jb.id,))
    try:
        job.init_storage()
        job.init_temp(jb.id)

        if not hasattr(job, 'label_file') or job.label_file is None:
            job.label_file = 'corpus/' + job.prep_name + "/object-detection.pbtxt"
        job.init_labels()
        if len(job.categories) == 0:
            raise ValueError(
                'no classes identified i {}'.format(job.label_file))
        job.jb = jb
        source = json.loads(job.download_to_string(
            'corpus/' + job.prep_name + "/job_def.json"))
        job.project_name = source['project_name']
        jb.meta['steps'] = 4
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'import_config'
        jb.meta['current_step'] = 0
        jb.save_meta()
        download_training_data(job)
        updateFile(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'train'
        jb.meta['current_step'] += 1
        jb.save_meta()
        train(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'export'
        jb.meta['current_step'] += 1
        jb.save_meta()
        export(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'upload'
        jb.meta['current_step'] += 1
        jb.save_meta()
        upload_training(job)
        history = json.loads(job.download_to_string(
            'corpus/' + job.prep_name + "/job_history.json"))
        upload_metadata(job, "trained_models/" + job.name, histroy)
        upload_labels(job, 'trained_models/' + job.name)
        job.cleanup()
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'done'
        jb.meta['current_step'] += 1
        dt = datetime.now()
        jb.meta['job_end_time'] = str(int(dt.timestamp()*1000))
        jb.save_meta()
        job.upload_data(job.to_json_string(), 'jobs/finished/{}_{}_train_d_{}.json'.format(
            str(job.start_time), str(job.end_time), jb.id), contentType='application/json')

        return "done"
    except:
        var = traceback.format_exc()
        jb.meta['job_exception'] = var
        jb.save_meta()
        dt = datetime.now()
        job.end_time = int(dt.timestamp()*1000)
        job.exception = var
        try:
            job.upload_data(job.to_json_string(
            ), 'jobs/failed/{}_{}_train_f_{}.json'.format(str(job.start_time), str(job.end_time), jb.id), contentType='application/json')
        except:
            pass
        raise
    finally:
        try:
            ct = 'd'
            if hasattr(job,'exception'):
                ct = 'f'
            job.upload_data(job.to_json_string(
            ), 'jobs/all/{}_{}_train_{}_{}.json'.format(str(job.start_time), str(job.end_time), ct,jb.id), contentType='application/json')
        except:
            pass
        try:
            job.delete_cloud_file(
                'jobs/running/{}_0_train_r_{}.json'.format(str(job.start_time), jb.id))
        except:
            pass
        try:
            job.delete_cloud_file(
                'jobs/all/{}_0_train_r_{}.json'.format(str(job.start_time), jb.id))
        except:
            pass


def train_mlengine(job):
    jb = get_current_job()
    dt = datetime.now()
    jb.meta['job_exec_time'] = str(int(dt.timestamp()*1000))
    logger.debug('Current job: %s' % (jb.id,))
    try:
        job.init_storage()
        job.init_temp(jb.id)
        if hasattr(job, 'source_training') and job.source_training is not '':
            sjd = json.loads(job.download_to_string(
                'training_jobs/' + job.source_training + "/job_def.json"))
            job.num_train_steps += sjd['num_train_steps']
            job.model = sjd['model']
            st = 'training_jobs/{}/'.format(job.source_training)
            dt = 'training_jobs/{}/'.format(job.name)
            job.copy_folder(st, dt,'events.out.tfevents')
            job.delete_cloud_file('{}{}'.format(dt, "job_def.json"))
            job.delete_cloud_file('{}{}'.format(dt, "job_history.json"))
        if not hasattr(job, 'label_file') or job.label_file is None:
            job.label_file = 'corpus/' + job.prep_name + "/object-detection.pbtxt"
        job.init_labels()
        if len(job.categories) == 0:
            raise ValueError(
                'no classes identified i {}'.format(job.label_file))

        job.jb = jb
        source = json.loads(job.download_to_string(
            'corpus/' + job.prep_name + "/job_def.json"))
        job.project_name = source['project_name']
        job.num_evals = source['test_image_ct']-1
        jb.meta['steps'] = 6
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'preping_config'
        jb.meta['current_step'] = 0
        jb.save_meta()
        updateFileML(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'uploading_model'
        jb.meta['current_step'] += 1
        jb.save_meta()
        upload_model(job)
        upload_packages(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'start_ml_engine'
        jb.meta['current_step'] += 1
        jb.save_meta()
        job.tr_result = start_ml_engine(job)
        history = json.loads(job.download_to_string(
            'corpus/' + job.prep_name + "/job_history.json"))
        upload_metadata(job, "training_jobs/" + job.name, history)
        if job.tr_result == 'SUCCEEDED':
            job.training = job.name
            job.name = job.project_name
            job.deploy_ml_engine = False
            jb.meta['current_step_processed'] = 0
            jb.meta['current_step_size'] = 0
            jb.meta['current_step_name'] = 'download_training'
            jb.meta['current_step'] += 1
            jb.save_meta()
            download_eval(job)
            download_training(job)
            jb.meta['current_step_processed'] = 0
            jb.meta['current_step_size'] = 0
            jb.meta['current_step_name'] = 'export'
            jb.meta['current_step'] += 1
            jb.save_meta()
            export(job, input_type='encoded_image_string_tensor')
            jb.meta['current_step_processed'] = 0
            jb.meta['current_step_size'] = 0
            jb.meta['current_step_name'] = 'upload'
            jb.meta['current_step'] += 1
            jb.save_meta()
            upload_training(job)
            jb.meta['current_step_processed'] = 0
            jb.meta['current_step_size'] = 0
            jb.meta['current_step_name'] = 'deploy'
            jb.meta['current_step'] += 1
            jb.save_meta()
            if not hasattr(job, 'deploy_ml_engine') or job.deploy_ml_engine:
                deploy_ml_engine(job)
                jb.meta['current_step_processed'] = 0
                jb.meta['current_step_size'] = 0
                jb.meta['current_step_name'] = 'upload_metadata'
                jb.meta['current_step'] += 1
                jb.save_meta()
            history = json.loads(job.download_to_string(
                'training_jobs/' + job.training + "/job_history.json"))
            upload_metadata(job, 'trained_models/' + job.name +
                            '/' + str(job.model_version), history)
            upload_labels(job, 'trained_models/' + job.name +
                          '/' + str(job.model_version))
            create_model_pbtxt(job)
        job.cleanup()
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'done'
        jb.meta['current_step'] += 1
        dt = datetime.now()
        jb.meta['job_end_time'] = str(int(dt.timestamp()*1000))
        jb.save_meta()
        job.upload_data(job.to_json_string(
        ), 'jobs/finished/{}_{}_train_d_{}.json'.format(str(job.start_time), str(job.end_time), jb.id), contentType='application/json')

        return job
    except:
        var = traceback.format_exc()
        jb.meta['job_exception'] = var
        dt = datetime.now()
        job.end_time = int(dt.timestamp()*1000)
        jb.save_meta()
        job.exception = var
        try:
            job.delete_cloud_file('jobs/running/{}.json'.format(jb.id))
            job.upload_data(job.to_json_string(
            ), 'jobs/failed/{}_{}_train_d_{}.json'.format(str(job.start_time), str(job.end_time), jb.id), contentType='application/json')
        except:
            pass
        raise
    finally:
        try:
            ct = 'd'
            if hasattr(job,'exception'):
                ct = 'f'
            job.upload_data(job.to_json_string(
            ), 'jobs/all/{}_{}_train_{}_{}.json'.format(str(job.start_time), str(job.end_time),ct, jb.id), contentType='application/json')
        except:
            pass
        try:
            job.delete_cloud_file(
                'jobs/running/{}_0_train_r_{}.json'.format(str(job.start_time), jb.id))
        except:
            pass
        try:
            job.delete_cloud_file(
                'jobs/all/{}_0_train_r_{}.json'.format(str(job.start_time), jb.id))
        except:
            pass


def export_mlengine(job):
    jb = get_current_job()
    dt = datetime.now()
    jb.meta['job_exec_time'] = str(int(dt.timestamp()*1000))
    logger.debug('Current job: %s' % (jb.id,))
    try:
        job.init_storage()
        job.init_temp(jb.id)
        if not hasattr(job, 'label_file') or job.label_file is None:
            job.label_file = 'training_jobs/'+job.training + "/object-detection.pbtxt"
        source = json.loads(job.download_to_string(
            'training_jobs/' + job.training + "/job_def.json"))
        job.project_name = source['project_name']
        job.init_labels()
        job.jb = jb
        jb.meta['steps'] = 3
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'download_training'
        jb.meta['current_step'] = 0
        jb.save_meta()
        download_training(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'export'
        jb.meta['current_step'] += 1
        jb.save_meta()
        export(job, input_type='encoded_image_string_tensor')
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'upload'
        jb.meta['current_step'] += 1
        jb.save_meta()
        upload_training(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'deploy'
        jb.meta['current_step'] += 1
        jb.save_meta()
        if not hasattr(job, 'deploy_ml_engine') or job.deploy_ml_engine:
            deploy_ml_engine(job)
            jb.meta['current_step_processed'] = 0
            jb.meta['current_step_size'] = 0
            jb.meta['current_step_name'] = 'upload_metadata'
            jb.meta['current_step'] += 1
            jb.save_meta()
        history = json.loads(job.download_to_string(
            'training_jobs/' + job.training + "/job_history.json"))
        upload_metadata(job, 'trained_models/' + job.name +
                        '/' + str(job.model_version), history)
        upload_labels(job, 'trained_models/' + job.name +
                      '/' + str(job.model_version))
        create_model_pbtxt(job)
        job.cleanup()
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'done'
        jb.meta['current_step'] += 1
        dt = datetime.now()
        jb.meta['job_end_time'] = str(int(dt.timestamp()*1000))
        jb.save_meta()
        job.upload_data(job.to_json_string(
        ), 'jobs/finished/{}_{}_export_d_{}.json'.format(str(job.start_time), str(job.end_time), jb.id), contentType='application/json')

        return job
    except:
        var = traceback.format_exc()
        dt = datetime.now()
        job.end_time = int(dt.timestamp()*1000)
        jb.meta['job_exception'] = var
        jb.save_meta()
        job.exception = var
        try:
            job.upload_data(job.to_json_string(
            ), 'jobs/failed/{}_{}_export_f_{}.json'.format(str(job.start_time), str(job.end_time), jb.id), contentType='application/json')
        except:
            pass
        raise
    finally:
        try:
            ct = 'd'
            if hasattr(job,'exception'):
                ct = 'f'
            job.upload_data(job.to_json_string(
            ), 'jobs/all/{}_{}_export_{}_{}.json'.format(str(job.start_time), str(job.end_time), ct,jb.id), contentType='application/json')
        except:
            pass
        try:
            job.delete_cloud_file(
                'jobs/running/{}_0_export_r_{}.json'.format(str(job.start_time), jb.id))
        except:
            pass
        try:
            job.delete_cloud_file(
                'jobs/all/{}_0_export_r_{}.json'.format(str(job.start_time), jb.id))
        except:
            pass
