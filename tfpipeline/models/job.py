
import shutil
import os
import tempfile
from pathlib import Path
base_dir='./'
from google.protobuf import text_format
import utils.string_int_label_map_pb2 as labelmap
from google.cloud import storage
from google.cloud.storage import Blob
import utils.storage

import boto3
import botocore

import json


class Job(object):
    def __init__(self, name, parameters):
        self.__dict__ = parameters
        self.name = name

        if not 'use_gcs' in parameters:
            self.use_gcs =False

        if not 'train_samples' in parameters:
            self.train_samples = 100
        if not 'test_samples' in parameters:
            self.test_samples = 10
        if 'bucket' not in parameters:
            self.bucket = os.environ['BUCKET']
        print("done")

    def init_s3(self):
        state = dict(self.__dict__)
        if 'access_key' not in state:
            self.access_key = os.environ['ACCESS_KEY']
        if 'secret_key' not in state:
            self.secret_key = os.environ['SECRET_KEY']
        if 'url' not in state:
            self.url = os.environ['S3_URL']

        self.s3 = boto3.client('s3',
                               endpoint_url= self.url,
                               config=boto3.session.Config(signature_version='s3v4'),
                               aws_access_key_id=self.access_key,
                               aws_secret_access_key=self.secret_key
                               )
        if hasattr(self,'project_name'):
            _obj = self.s3.get_object(Bucket=self.bucket,Key='projects/{}/state.json'.format(self.project_name))
            jsonbody = str(_obj['Body'].read().decode('utf-8'))
            self.project = json.loads(jsonbody)
    
    def init_storage(self):
        if self.use_gcs:
            self.init_gcs()
        else:
            self.init_s3()

    def init_gcs(self):
        self.gcs = storage.Client()
        self.gcs_bucket = self.gcs.get_bucket(self.bucket)

        if hasattr(self,'project_name'):
            jsonbody = self.download_to_string('projects/{}/state.json'.format(self.project_name))
            self.project = json.loads(jsonbody)
            

    def init_temp(self,uuid):
        try:
            shutil.rmtree(tempfile.gettempdir() +'/' + uuid)
        except:
            print('temp dir ' + tempfile.gettempdir() +'/' + uuid + ' does not exist')
        tmpdir = Path(tempfile.gettempdir() +'/' + uuid)
        if not tmpdir.is_dir():
            os.mkdir(tempfile.gettempdir() +'/' +  uuid)
        self.tempdir = tempfile.gettempdir() +'/' +  uuid

    def init_labels(self):
        if not hasattr(self, 'label_file') or self.label_file is None:
            self.label_map ={}
            self.labelmap_dict = {}
            self.categories = []
        else:
            self.get_classes()

    def cleanup(self):
        shutil.rmtree(self.tempdir)

    def delete_cloud_file(self,file):
        if self.use_gcs:
            
            blob = self.gcs_bucket.get_blob(file)
            blob.delete()
        else:
            self.s3.delete_object(Bucket=self.bucket,Key=file)
    
    def copy_object(self,source,dest):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            print('Size: {} bytes'.format(blob.size))
            self.gcs_bucket.copy_blob(blob,self.gcs_bucket,dest)
        else:
            self.s3.copy_object(Bucket=self.bucket, CopySource=self.bucket + '/'+ source, Key=dest)
    
    def copy_folder(self,source,dest,ignoreString=None):
        if not source.endswith("/"):
            source = source + "/"
        if not dest.endswith("/"):
            dest = dest + "/"
        if self.use_gcs:
            blobs = self.gcs_bucket.list_blobs(prefix=source)
            for blob in blobs:
                df = "{}{}".format(dest,blob.name[len(source):])
                if ignoreString is None or blob.name.find(ignoreString) == -1:
                    self.gcs_bucket.copy_blob(blob,self.gcs_bucket,df)
                blob = self.gcs_bucket.get_blob(source)
            
        else:
            while True:
                kwargs = {'Bucket': self.bucket,
                'Prefix':source}
                resp = self.s3.list_objects_v2(**kwargs)
                try:
                    for obj in resp['Contents']:
                        df = "{}{}".format(dest,obj['Key'][len(source):])
                        if ignoreString is None or blob.name.find(ignoreString) == -1:
                            self.s3.copy_object(Bucket=self.bucket, CopySource=self.bucket + '/'+ source, Key=df)
                except KeyError:
                    break
                try:
                    kwargs['ContinuationToken'] = resp['NextContinuationToken']
                except KeyError:
                    break
    
    def list_files(self,prefix,delimiter=None,metadata=False):
        lst = []
        if self.use_gcs:
            blobs = self.gcs_bucket.list_blobs(prefix=prefix, delimiter=delimiter)
            for blob in blobs:
                if metadata:
                    lst.append({
                        'name': blob.name,
                        'ETag': blob.etag,
                        'size': blob.size,
                        'lastModified': int(blob.updated.timestamp()*1000)

                    })
                else:
                    lst.append(blob.name)
        else:
            while True:
                kwargs = {'Bucket': self.bucket,
                'Prefix':prefix}
                if delimiter is not None:
                    kwargs['Delimiter'] = delimiter
                resp = self.s3.list_objects_v2(**kwargs)
                try:
                    
                    for obj in resp['Contents']:
                        if metadata:
                            lst.append({
                                'name': obj['Key'],
                                'ETag': obj['ETag'].replace('"', ''),
                                'size': obj['Size'],
                                'lastModified': int(obj['LastModified'].timestamp()*1000)

                            })
                        else:
                            lst.append(obj['Key'])
                        
                except KeyError:
                    break
                try:
                    kwargs['ContinuationToken'] = resp['NextContinuationToken']
                except KeyError:
                    break
        return lst
    
    def upload_file(self,source,dest,contentType=None):
        if self.use_gcs:
            args = {}
            if not contentType is None:
                args ={'content_type':contentType}
            blob = Blob(dest, self.gcs_bucket)
            blob.upload_from_filename(source,**args)
        else:
            args = {}
            if not contentType is None:
                args ={'ContentType':contentType}
            self.s3.upload_file(source,self.bucket, dest,ExtraArgs=args)

    def upload_data(self,data,dest,contentType=None):
        if self.use_gcs:
            args = {}
            if not contentType is None:
                args = {'content_type' : contentType}
            blob = Blob(dest, self.gcs_bucket)
            blob.upload_from_string(data,**args)
        else:
            args = {
                'Body' : data,
                'Bucket' : self.bucket,
                'Key' : dest,
            }
            if not contentType is None:
                args['ContentType'] = contentType
            self.s3.put_object(**args)

    def download_to_file(self,source,dest):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            blob.download_to_filename(dest)
        else:
            self.s3.download_file(self.bucket, source, dest)

    def download_to_data(self,source):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            data = blob.download_as_string()
            return data
        else:    
            data = self.s3.get_object(Bucket=self.bucket,Key=source)['Body'].read()
            return data
    
    def download_to_string(self,source):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            data = str(blob.download_as_string().decode('utf-8'))
            return data
        else:    
            data = str(self.s3.get_object(Bucket=self.bucket,Key=source)['Body'].read().decode('utf-8'))
            return data

    def exists(self,source):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            if blob:
                return blob.exists()
            else:
                return False
        else:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=source)
                return True
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
                else:
                    print(e)
                    raise ValueError('Error getting ' +source + ' in ' + self.bucket )

    def __getstate__(self):
        state = dict(self.__dict__)
        if 's3' in state:
            del state['s3']
        if 'gcs' in state:
            del state['gcs']
        if 'gcs_bucket' in state:
            del state['gcs_bucket']
        if 'label_map' in state:
            del state['label_map']
        return state

    def to_export(self):
        state = dict(self.__dict__)
        if 's3' in state:
            del state['s3']
        if 'gcs' in state:
            del state['gcs']
        if 'gcs_bucket' in state:
            del state['gcs_bucket']
        if 'access_key' in state:
            del state['access_key']
        if 'secret_key' in state:
            del state['secret_key']
        if 'jb' in state:
            del state['jb']
        if 'label_map' in state:
            del state['label_map']
        if 'project' in state:
            del state['project']
        if 'traincoco' in state:
            del state['traincoco']
        if 'testcoco' in state:
            del state['testcoco']
        return state
    
    def to_json_string(self):
        return json.dumps(self.to_export())

    def get_classes(self):
        try:
            label_map_string = ''
            print("getting "+ self.label_file + ' from ' + self.bucket)
            if self.use_gcs:
                blob = self.gcs_bucket.get_blob(self.label_file)
                label_map_string = blob.download_as_string().decode('utf-8')
            else:
                obj = self.s3.get_object(Bucket=self.bucket,Key=self.label_file)
                label_map_string = obj['Body'].read().decode('utf-8')
            print("got "+ self.label_file)
            self.label_map ={}
            self.label_map = labelmap.StringIntLabelMap()
            self.labelmap_dict = {}
            text_format.Merge(label_map_string, self.label_map)
            self.categories = []
            for item in self.label_map.item:
              self.labelmap_dict[item.name] = item.id
              self.categories.append({'id': item.id, 'name': item.name})
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                self.labelmap_dict = {}
                print(self.label_file + ' was not found')
            else:
                print(e)
                raise ValueError('Error getting ' +self.label_file + ' in ' + self.bucket )
