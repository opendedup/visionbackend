from google.cloud import storage
from google.cloud.storage import Blob
from google.cloud.exceptions import NotFound

import os
import logging

import boto3
import botocore

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
from googleapiclient.errors import HttpError



class Storage(object):
    def __init__(self,parameters):
        if 'ACCESS_KEY' in parameters:
            self.access_key = parameters['ACCESS_KEY']
        if 'SECRET_KEY' in parameters:
            self.secret_key = parameters['SECRET_KEY']
        if 'S3_URL' in parameters:
            self.url = parameters['S3_URL']
        if 'BUCKET' in parameters:
            self.bucket = parameters['BUCKET']
        if 'USE_GCS' in parameters and parameters['USE_GCS'] == True:
            self.use_gcs = True
        else:
            self.use_gcs = False
        if self.use_gcs:
            self.gcs = storage.Client()
            self.sclient = discovery.build('storage', 'v1') # add http=whatever param if auth
            try:
                self.gcs_bucket = self.gcs.get_bucket(self.bucket)

            except google.cloud.exceptions.NotFound:
                self.gcs_bucket = self.gcs.create_bucket(self.bucket)
        else:
            self.s3 = boto3.client('s3',
                                endpoint_url= self.url,
                                config=boto3.session.Config(signature_version='s3v4'),
                                aws_access_key_id=self.access_key,
                                aws_secret_access_key=self.secret_key
                                )
            try:
                self.s3.head_bucket(Bucket=self.bucket)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    self.s3.create_bucket(Bucket=self.bucket)
                else:
                    logging.error(e)
                    raise ValueError('Error initializing ' + self.bucket )
                
        
    def list_files(self,prefix,delimiter=None,metadata=False,folders_only=False):
        lst = []
        if self.use_gcs:
            if folders_only:
                request = self.sclient.objects().list(
                bucket=self.bucket,
                prefix=prefix,
                delimiter=delimiter)
                response = request.execute()
                for s in response['prefixes']:
                    lst.append(s)
            else:        
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
                    if folders_only:
                        for obj in resp['CommonPrefixes']:
                            lst.append(obj['Prefix'])
                    else:
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
    
    def delete_cloud_dir(self,folder):
        lst = self.list_files(folder)
        for l in lst:
            self.delete_cloud_file(l)

    

    def delete_cloud_file(self,file):
        if self.use_gcs:
            
            blob = self.gcs_bucket.get_blob(file)
            blob.delete()
        else:
            self.s3.delete_object(Bucket=self.bucket,Key=file)
    
    def copy_object(self,source,dest):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            self.gcs_bucket.copy_blob(blob,self.gcs_bucket,dest)
        else:
            self.s3.copy_object(Bucket=self.bucket, CopySource=self.bucket + '/'+ source, Key=dest)
    
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

    def upload_path(self,local_directory,dest):
        for root, _, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(dest, relative_path)
                self.upload_file(local_path, s3_path)
    
    def exists(self,source):
        if self.use_gcs:
            blob = self.gcs_bucket.get_blob(source)
            return blob.exists()
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


        
