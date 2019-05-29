import sys
import logging
from object_detection.dataset_tools import tf_record_creation_util
import contextlib2
from datetime import datetime
import threading
from models.job import Job
import uuid
from rq import get_current_job
import urllib3
import multiprocessing
import concurrent.futures
from collections import namedtuple, OrderedDict
from object_detection.utils import dataset_util
from PIL import Image
import cv2
import boto3
import traceback
import utils.string_int_label_map_pb2 as labelmap
from google.protobuf import text_format
import json
import tensorflow as tf
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import imageio
from imgaug import augmenters as iaa
import imgaug as ia
import os
import io
import tempfile
from pathlib import Path
import random
base_dir = './'


http = urllib3.PoolManager()
imcont = 0
objct = 0


global_lock = threading.Lock()


logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def class_text_to_int(job, row_label):
    id = job.labelmap_dict[row_label]
    return id


def create_label_pbtxt(job):
    out_file = job.tempdir + '/object-detection.pbtxt'
    with open(out_file, "w") as fout:
        for key, value in job.labelmap_dict.items():
            fout.write('item {\n'+'\tid: ' + str(value) +
                       '\n\tname: \''+key+'\'\n}\n')
    job.upload_file(out_file, 'corpus/'+job.name + "/" +
                    os.path.basename(out_file), contentType='text/plain')


def upload_metadata(job):
    dt = datetime.now()
    job.end_time = int(dt.timestamp()*1000)
    history = {'project': job.project, 'history': [job.to_export()]}
    job.upload_data(json.dumps(history), 'corpus/'+job.name +
                    "/job_history.json", contentType='application/json')
    job.upload_data(job.to_json_string(), 'corpus/'+job.name +
                    "/job_def.json", contentType='application/json')
    job.upload_data(json.dumps(job.traincoco),'corpus/'+job.name +
                    "/traincoco.json", contentType='application/json')
    job.upload_data(json.dumps(job.testcoco),'corpus/'+job.name +
                    "/testcoco.json", contentType='application/json')


def process_group(job, group, ct_dict,train):
    cur_image = job.download_to_data(group.filename)
    image = Image.open(io.BytesIO(cur_image))
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    ct = 0
    ot = 0
    global imcont
    global objct
    with global_lock:
        imcont += 1
        objct +=1
        ot = objct
        ct = imcont
    
    for _, row in group.object.iterrows():
        annotation ={}
        annotation["iscrowd"] = 0
        annotation["image_id"] = ct

        wd = (row['xmax'] / width)-(row['xmin'] / width)
        ht = (row['ymax'] / height)-(row['ymin'] / height)
        if wd <= 0 or ht <= 0:
            print("width=" + str(wd) + " ht="+str(ht))
            return None
        x1 = row['xmin']
        y1 = row['ymin']
        x2 = row['xmax'] - x1
        y2 = row['ymax'] - y1
        annotation["bbox"] = [x1, y1, x2, y2]
        annotation["area"] = float(x2 * y2)
        annotation["category_id"] = class_text_to_int(job, row['class'])
        annotation["ignore"] = 0
        annotation["id"] = ot
        annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
        if train:
            job.traincoco["annotations"].append(annotation)
        else:
            job.testcoco["annotations"].append(annotation)
        xmin = float(row['xmin']) / width
        if xmin < 0:
            xmin = 0
        xmax = float(row['xmax']) / width
        if xmax > 1:
            print("xmax="+xmax)
            xmax = 1
        ymin = float(row['ymin']) / height
        if ymin < 0:
            ymin=0
        ymax = float(row['ymax']) / height
        if ymax > 1:
            print('ymax='+ymax)
            ymax =1

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        class_id = class_text_to_int(job, row['class'])
        classes_text.append(row['class'].encode('utf8'))
        with global_lock:
            if row['class'] not in ct_dict:
                if train:
                    job.traincoco['categories'].append({"supercategory": "none","id": class_id,"name": row['class']})
                else:
                    job.testcoco['categories'].append({"supercategory": "none","id": class_id,"name": row['class']})
                ct_dict[row['class']] = {'count': 1, 'id': class_id}
            else:
                ct_dict[row['class']]['count'] += 1
        
        classes.append(class_id)
    if len(classes) == 0 or len(classes) != len(xmins) or len(classes) != len(xmaxs) or len(classes) != len(ymins) or len(classes) != len(ymaxs):
        return None
    else:
        dt = datetime.now()
        img = {
            "license": 4,
            "file_name": group.filename,
            "height": height,
            "width": width,
            "date_captured": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "id": ct
        }
        if train:
            job.traincoco["images"].append(img)
        else:
            job.testcoco["images"].append(img)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/class/label': dataset_util.int64_feature(-1),
            'image/class/text': dataset_util.bytes_feature(str('').encode('utf8')),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(str(ct).encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(cur_image),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        # with global_lock:
        #    writer.write(tf_example.SerializeToString())
        return tf_example.SerializeToString()


def create_tf_example(job, train=True):
    csv_file = job.train_csv_file
    out_file = job.tempdir + '/train/train-record'
    if train:
        local_directory = job.tempdir + '/train/'
        os.mkdir(job.tempdir + '/train/')
    else:
        os.mkdir(job.tempdir + '/test/')
        local_directory = job.tempdir + '/test/'
        csv_file = job.test_csv_file
        out_file = job.tempdir + '/test/validation-record'
    local_csv = job.tempdir + '/' + str(uuid.uuid4()) + '.csv'
    job.download_to_file(csv_file, local_csv)
    if train:
        job.train_count_dict = {}
        ct_dict = job.train_count_dict
    else:
        job.test_count_dict = {}
        ct_dict = job.test_count_dict
    examples = pd.read_csv(local_csv)
    grouped = split(examples, 'filename')
    num_shards = 10
    output_filebase = out_file
    index = 1
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
            future_img = {executor.submit(
                process_group, job, group, ct_dict,train): group for group in grouped}
            for future in concurrent.futures.as_completed(future_img):
                #img = future_img[future]
                try:
                    data = future.result()
                    if data is not None:
                        with global_lock:
                            current_index = index
                            index += 1
                            output_shard_index = current_index % num_shards
                            output_tfrecords[output_shard_index].write(data)
                except Exception as exc:
                    print('%r generated an exception: %s' %
                          ("create_tf_example", exc))
                    var = traceback.format_exc()
                    print(var)
    file_ar = []
    if train:
        job.train_image_ct = index
    else:
        job.test_image_ct = index
    for root, dirs, files in os.walk(local_directory):

        for filename in files:

            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)

            s3_path = os.path.join('corpus/' + job.name + "/", relative_path)

            # relative_path = os.path.relpath(os.path.join(root, filename))
            #print("Uploading {} to {}...".format(local_path,s3_path))
            #tc = boto3.s3.transfer.TransferConfig()
            # t = boto3.s3.transfer.S3Transfer( client=job,
            #                             config=tc )
            #t.upload_file( filename, 'my-bucket-name', 'name-in-s3.dat' )
            file_ar.append({'local_path': local_path, 's3_path': s3_path})

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_up = {executor.submit(
            job.upload_file, obj['local_path'], obj['s3_path']): obj for obj in file_ar}
        for future in concurrent.futures.as_completed(future_up):
            img = future_up[future]
            try:
                data = future.result()
            except Exception as exc:
                print('upload of %r generated an exception: %s' %
                      (str(obj), exc))

    if train:
        job.train_record = 'corpus/' + job.name + "/train-record-?????-of-00010"
    else:
        job.test_record = 'corpus/' + job.name + "/validation-record-?????-of-00010"

    return 'corpus/' + job.name + "/" + os.path.basename(out_file)


def count_objects(job):
    kwargs = {'Bucket': job.bucket,
              'Prefix': job.class_folder}
    sz = 0
    while True:
        resp = job.s3.list_objects_v2(**kwargs)

        try:
            for obj in resp['Contents']:
                sz += 1
        except KeyError:
            break
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
    return sz


def process_children(parent, ar):
    for child in parent:
        if 'children' in child and len(child['children']) > 0:
            process_children(child, ar)
        ch = {}
        ch['xmin'] = child['shape']['p1']['x']
        ch['ymin'] = child['shape']['p1']['y']
        ch['xmax'] = child['shape']['p2']['x']
        ch['ymax'] = child['shape']['p2']['y']
        ch['tag'] = child['title'] + '_' + child['flag']
        if 'tool' in child:
            ch['tag'] = ch['tag'] + '_' + child['tool']
        else:
            ch['tag'] = ch['tag'] + '_none'
        ar.append(ch)


def process_json(job):
    train_csv_file = job.tempdir + '/train.csv'
    statefile = 'projects/'+job.project['id'] + '/state.json'
    jsonbody = job.download_to_string(statefile)
    state = json.loads(jsonbody)
    if hasattr(job, 'jb'):
        job.jb.meta['current_step_size'] = len(state['photos'])
        job.jb.meta['current_step_processed'] = 0
        job.jb.save_meta()
    trn_sz = len(state['photos'])
    test_pct_set = False
    if hasattr(job,"test_pct") and job.test_pct > 0:
        trn_sz = int((1-job.test_pct)*len(state['photos']))
        test_pct_set = True
    ct = 0
    for photo in state['photos']:
        if 'children' in photo and len(photo['children']) > 0:
            tags = []
            process_children(photo['children'], tags)
            fotopath = 'projects/' + photo['imgUrl']
            logger.debug(" looking for " + fotopath)
            bbox = []
            data = job.download_to_data(fotopath)
            img_array = np.array(bytearray(data), dtype=np.uint8)
            im = cv2.imdecode(img_array, -1)
            if hasattr(job, 'desired_size') and job.desired_size > 0:
                # old_size is in (height, width) format
                old_size = im.shape[:2]
                ratio = float(job.desired_size)/max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])

                # new_size should be in (width, height) format

                im = cv2.resize(im, (new_size[1], new_size[0]))

                delta_w = job.desired_size - new_size[1]
                delta_h = job.desired_size - new_size[0]
                top, bottom = 0, delta_h
                left, right = 0, delta_w

                color = [0, 0, 0]
                im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)
            # print(im.shape)
            for ch in tags:
                xmin = float(ch['xmin']*im.shape[1])
                ymin = float(ch['ymin']*im.shape[0])
                xmax = float(ch['xmax']*im.shape[1])
                ymax = float(ch['ymax']*im.shape[0])
                name = ch['tag']
                if name not in job.labelmap_dict:
                    job.labelmap_dict[name] = len(job.labelmap_dict)+1
                bbox.append(ia.BoundingBox(x1=xmin, y1=ymin,
                                           x2=xmax, y2=ymax, label=name))
            bbs = ia.BoundingBoxesOnImage(bbox, shape=im.shape)
            fout = open(train_csv_file, "w")
            img_type = 'TRAIN'
            if ct > trn_sz:
                img_type = 'UNASSIGNED'
            tfile = fotopath.split('.')[0] + '.tcapture'
            timages = []
            if job.exists(tfile):
                tcapture = json.loads(job.download_to_string(tfile))
                timages = tcapture['images']
            for i in range(len(bbox)):
                after = bbox[i]
                fout.write(img_type + ',gs://{}/{}'.format(job.bucket,fotopath) +','+after.label + ',' + str(ch['xmin']) + ',' + str(ch['ymin']) +
                           ',,,' + str(ch['xmax']) + ',' + str(ch['ymax']) + ',,' + '\n')
                for tim in timages:
                    fout.write(img_type + ',gs://{}/{}'.format(job.bucket,tim['path']) +','+after.label + ',' + str(ch['xmin'])
                             + ',' + str(ch['ymin']) +
                            ',,,' + str(ch['xmax']) + ',' + str(ch['ymax']) + ',,' + '\n')
            if hasattr(job, 'aug') and job.aug:
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_img = {executor.submit(
                        create_image, im, job, bbs, fout,img_type): i for i in range(job.train_samples)}
                    for future in concurrent.futures.as_completed(future_img):
                        img = future_img[future]
                        try:
                            data = future.result()
                        except Exception as exc:
                            print(
                                'train augmentation generated an exception: %s' % (exc))
                if not test_pct_set:
                    fout.close()
                    fout = open(test_csv_file,"a")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                        future_img = {executor.submit(create_image,im, job,bbs, fout,img_type): i for i in range(job.test_samples)}
                        for future in concurrent.futures.as_completed(future_img):
                            img = future_img[future]
                            try:
                                data = future.result()
                            except Exception as exc:
                                print('test augmentation generated an exception: %s' % (exc))
            fout.close()
            ct += 1
            if hasattr(job, 'jb'):
                job.jb.meta['current_step_processed'] += 1
                job.jb.save_meta()

    print(job.labelmap_dict)
    job.upload_file(train_csv_file, 'corpus/' + job.name +
                    "/" + os.path.basename(train_csv_file))
    
    job.train_csv_file = 'corpus/' + job.name + \
        "/" + os.path.basename(train_csv_file)





def delete_staged(job):
    """Deletes all staged objects for the bucket"""
    kwargs = {'Bucket': job.bucket,
              'Prefix': 'corpus/' + job.name + '/stage'}

    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(
            prefix='corpus/' + job.name + '/stage')
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_up = {executor.submit(blob.delete): blob for blob in blobs}
            for future in concurrent.futures.as_completed(future_up):
                img = future_up[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (str(obj), exc))

    else:
        while True:
            resp = job.s3.list_objects_v2(**kwargs)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_up = {executor.submit(
                        job.s3.delete_object, Bucket=job.bucket, Key=obj['Key']): obj for obj in resp['Contents']}
                    for future in concurrent.futures.as_completed(future_up):
                        img = future_up[future]
                        try:
                            data = future.result()
                        except Exception as exc:
                            print('%r generated an exception: %s' %
                                  (str(obj), exc))

            except KeyError:
                break
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

    job.delete_cloud_file(job.train_csv_file)
    job.train_csv_file = None

def project_exists(job):
    key = 'corpus/'+job.name + '/'
    if job.use_gcs:
        blobs = job.gcs_bucket.list_blobs(prefix=key)
        for blob in blobs:
            print("corpus exists in bucket " + key + ' blob=' + blob.name)
            return True

    else:
        resp = job.s3.list_objects(Bucket=job.bucket, Prefix=key)
        if 'Contents' in resp and len(resp['Contents']) > 0:
            print("corpus exists in bucket " + key)
            return True
    return False


def create_image(im, job, bbs, fout,img_type):
    image_name = "{}/{}/stage/{}{}".format('corpus',
                                           job.name, uuid.uuid4(), '.jpg')
    aug_rules = {}
    if hasattr(job, 'aug_rules'):
        aug_rules = job.aug_rules

    affine = {
    }

    if 'scale' in aug_rules and aug_rules['scale']:
        affine['scale'] = (0.8, 1.0)

    if 'shear' in aug_rules and aug_rules['shear']:
        affine['shear'] = (-2, 2)

    if 'rotate' in aug_rules:
        rotate = aug_rules['rotate']
        affine['rotate'] = (rotate[0], rotate[1])

    ia.seed(random.randint(0, 1000000))
    aug = [
        iaa.Affine(**affine)
    ]
    if 'hflip' in aug_rules:
        aug.append(
            iaa.Fliplr(aug_rules['hflip'])
        )
    if 'vflip' in aug_rules:
        aug.append(
            iaa.Flipud(aug_rules['vflip'])
        )
    if 'contrast' in aug_rules and aug_rules['contrast']:
        aug.append(iaa.Multiply((0.8, 1.2)))
        aug.append(iaa.ContrastNormalization((0.8, 1.2)))

    if 'noise' in aug_rules and aug_rules['noise']:
        aug.append(iaa.Sometimes(0.5,
                                 iaa.GaussianBlur(sigma=(0, 0.5))
                                 ))
        aug.append(iaa.Sometimes(0.2,
                                 # randomly remove up to 10% of the pixels
                                 iaa.Dropout((0.01, 0.1), per_channel=0.5)
                                 ))
        aug.append(iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.05*255), per_channel=0.5))

    if 'crop' in aug_rules and aug_rules['crop']:
        aug.append(iaa.Crop(percent=(0, 0.1)))

    seq = iaa.Sequential(aug, random_order=True)
    seq_det = seq.to_deterministic()
    images_aug = seq_det.augment_image(im)
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    if 'remove_out_of_image' in aug_rules:
        rmi = aug_rules['remove_out_of_image']
        if rmi == 'cut_out_partly':
            bbs_aug = bbs_aug.remove_out_of_image(partly=True)
        elif rmi == 'cut_out_of_image':
            bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()
        elif rmi == 'leave_partly_in_image':
            bbs_aug = bbs_aug.remove_out_of_image()
    else:
        bbs_aug = bbs_aug.remove_out_of_image(partly=True)

    ret, img = cv2.imencode('.jpg', images_aug)
    job.upload_data(img.tobytes(), image_name, contentType='image/jpeg')
    #bbs = bbs_aug.remove_out_of_image().cut_out_of_image()

    with global_lock:
        for i in range(len(bbs_aug.bounding_boxes)):
            after = bbs_aug.bounding_boxes[i]
            xmin = float(after.x1/images_aug.shape[1])
            ymin = float(after.y1/images_aug.shape[0])
            xmax = float(after.x2/images_aug.shape[1])
            ymax = float(after.y2/images_aug.shape[0])
            fout.write(img_type+ ',gs://{}{}'.format(job.bucket,image_name) + ',' +
                       ','+after.label + ',' + str(xmin) + ',' + str(ymin) + ',,,' + str(xmax) + ',' + str(ymax) + ',,\n')
    return image_name


def preprocess(job):
    jb = get_current_job()
    print('Current job: %s' % (jb.id,))
    try:
        job.type = 'preprocess'
        job.init_storage()
        dt = datetime.now()
        jb.meta['job_exec_time'] = str(int(dt.timestamp()*1000))
        job.init_temp(jb.id)
        job.init_labels()
        job.jb = jb
        job.traincoco = {"info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2018,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
            "licenses": [],
            "images": [],
            "categories": [],
            "annotations": [],
        }
        job.testcoco = {"info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2018,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
            "licenses": [],
            "images": [],
            "categories": [],
            "annotations": [],
        }

        if hasattr(job, 'aug') and job.aug:
            jb.meta['steps'] = 6
        else:
            jb.meta['steps'] = 5
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_name'] = 'prep_existing_images'
        jb.meta['current_step'] = 0
        jb.save_meta()
        process_json(job)
        jb.meta['current_step_processed'] = 0
        jb.meta['current_step_size'] = 1
        jb.meta['current_step_name'] = 'create_tag_lables'
        jb.meta['current_step'] += 1

        create_label_pbtxt(job)
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'create_training_corpus'
        jb.meta['current_step'] += 1
        jb.save_meta()
        create_tf_example(job)
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'create_testing_corpus'
        jb.meta['current_step'] += 1
        jb.save_meta()
        create_tf_example(job, False)
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'cleaning_up'
        jb.meta['current_step'] += 1
        jb.save_meta()
        delete_staged(job)
        upload_metadata(job)
        jb.meta['current_step_size'] = 0
        jb.meta['current_step_name'] = 'done'
        jb.meta['current_step'] += 1
        dt = datetime.now()
        jb.meta['job_end_time'] = str(int(dt.timestamp()*1000))
        jb.save_meta()
        job.upload_data(job.to_json_string(),'jobs/finished/{}_{}_preprocess_d_{}.json'.format(str(job.start_time),str(job.end_time),jb.id),contentType='application/json')
        return job
    except:
        var = traceback.format_exc()
        dt = datetime.now()
        job.end_time = int(dt.timestamp()*1000)
        jb.meta['job_exception'] = var
        job.exception=var
        try:
            job.upload_data(job.to_json_string(),'jobs/failed/{}_{}_preprocess_f_{}.json'.format(str(job.start_time),str(job.end_time),jb.id),contentType='application/json')
        except:
            pass
        jb.save_meta()
        raise
    finally:
        try:
            ct = 'd'
            if hasattr(job,'exception'):
                ct = 'f'
            job.upload_data(job.to_json_string(),'jobs/all/{}_{}_preprocess_{}_{}.json'.format(str(job.start_time),str(job.end_time),ct,jb.id),contentType='application/json')
        except:
            pass
        try:
            
            job.delete_cloud_file('jobs/running/{}_0_preprocess_r_{}.json'.format(str(job.start_time),jb.id))
        except:
            pass
        try:
            job.delete_cloud_file('jobs/all/{}_0_preprocess_r_{}.json'.format(str(job.start_time),jb.id))
        except:
            pass
        job.cleanup()

