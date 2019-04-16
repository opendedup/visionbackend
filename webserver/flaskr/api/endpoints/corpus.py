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

import settings

import concurrent.futures
import multiprocessing

log = logging.getLogger(__name__)


ns = Namespace('corpus', description='Corpus used for training')

category_count = ns.model('Category', {
    'id': fields.Integer(required=True, description='The index id for the tag',
                          example=1),
    'name': fields.String(required=True, description='The category name',
                          example='Pins'),
    'count': fields.Integer(required=True, description='Number of instances found in corpus',
                          example=1),
})

aug_rules = ns.model('Augmentation_Rules', {
    'remove_out_of_image': fields.String(required=False,description="""Specify "cut_out_partly" to cut out objects that are partly out of view.
    Specify "cut_out_of_image" to trim the box to the image field of view. Specify "leave_partly_in_image" to keep all boxes regardless of their field of view.
     """,example="cut_out_of_image",
     enum=['leave_partly_in_image',
                                 'cut_out_partly','cut_out_of_image'],
                              default="cut_out_of_image"),
    'hflip':fields.Float(required=False, description='Horizontally Flip and Image a designated percentage of the time.',
                          default=0,
                          example=.5),
    'vflip':fields.Float(required=False, description='Vertically Flip and Image a designated percentage of the time.',
                          default=0,
                          example=.5),
    'rotate':fields.List(fields.Integer,required=False,description='Rotate objects clockwise and counterclockwise',
                          max_items=2,min_items=2,example=[90,-90]),
    'scale':fields.Boolean(required=False,description='Randomly add slightly scale the image randomly between 0.8x - 1x. ',
                              default=False),
    'shear':fields.Boolean(required=False,description='Randomly Slightly shear the image. ',
                              default=False),
    'contrast':fields.Boolean(required=False,description='Randomly add contrast to image. ',
                              default=False),
    'noise':fields.Boolean(required=False,description='Randomly add noise to images. ',
                              default=False),
    'crop':fields.Boolean(required=False,description='Randomly crop images. ',
                              default=False)
})

corpus = ns.model('Corpus', {
    'test_samples': fields.Integer(required=True, description='The number of samples used for testing the model',
                          example=100),
    'train_samples': fields.Integer(required=True, description='The number of samples used for training the model',
                          example=1000),
    'training':fields.String(required=True, description='The corpus used to train the model',
                          example='data1'),
    'bucket':fields.String(required=True, description='The bucket where the data is located',
                          example='bucket0'),
    'name': fields.String(required=True, description='The model name',
                          example='model0'),
    'project_name': fields.String(required=True, description='The name of the project assigned to this training',
                          example='project0'),
    'train_count_dict': fields.List(fields.Nested(category_count,required=False, description='Items found creating training corpus')),
    'test_count_dict': fields.List(fields.Nested(category_count,required=False, description='Items found creating testing corpus')),
    'aug_rules': fields.Nested(aug_rules,required=False, description='Image Augmentation rules used for image augmentation'),
    'end_time' :fields.Integer(required=False, description='The end time of the corpus creation as a Unix Timestamp',
                          example=1551641464731)
})


def get_cat(cat):
    ar = []
    for key in cat.keys():
        mm = {
            'name':key,
            'id':cat[key]['id'],
            'count':cat[key]['count']
        }
        ar.append(mm)
    return ar

def download_jobdef(file,mp):
    fldr = 'corpus/'
    try:
        jd = settings.storage.download_to_string(file + 'job_def.json')
        tl = file[len(fldr):]
        t = tl.split('/')
        mm = json.loads(jd)
        mm['name']=t[0]
        mm['bucket'] = settings.data['BUCKET']
        mm['train_count_dict'] = get_cat(mm['train_count_dict'])
        mm['test_count_dict'] = get_cat(mm['test_count_dict'])
        mp[t[0]]=mm
    except:
        pass

@ns.route('/')
class CorpusCollection(Resource):
    @ns.marshal_list_with(corpus)
    def get(self):
        """
        Returns list of Corpus.
        """
        fldr = 'corpus/'
        lst = settings.storage.list_files(fldr,delimiter='/',folders_only=True)
        mp = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
            future_img = {executor.submit(
                download_jobdef, l, mp): l for l in lst}
            for future in concurrent.futures.as_completed(future_img):
                #img = future_img[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' %
                          ("create_tf_example", exc))
                    var = traceback.format_exc()
                    log.error(var)    
        return list(mp.values())

@ns.route('/<string:name>')
@ns.response(404, 'Corpus not found.')
class CorpusItem(Resource):

    @ns.marshal_with(corpus)
    def get(self, name):
        """
        Returns a corpus of data.
        """
        fl = 'corpus/' + name + '/job_def.json'
        if not settings.storage.exists(fl):
            return 'Corpus not found',404
        else:
            jd = settings.storage.download_to_string(fl)
            mm = json.loads(jd)
            mm['name']=name
            mm['bucket'] = settings.data['BUCKET']
            mm['train_count_dict'] = get_cat(mm['train_count_dict'])
            mm['test_count_dict'] = get_cat(mm['test_count_dict'])
            return mm
    
    @ns.response(204, 'Model successfully deleted.')
    def delete(self, name):
        """
        Deletes a corpus.
        """
        fl = 'corpus/' + name + '/job_def.json'
        if not settings.storage.exists(fl):
            return 'Corpus not found',404
        else:
            settings.storage.delete_cloud_dir("corpus/{}/".format(name))
        return None, 204
       