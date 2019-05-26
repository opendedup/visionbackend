
import cv2
import atexit

import platform
print(platform.python_version())
import numpy as np
import os
import six.moves.urllib as urllib
import sys
sys.path.append('flaskr')
import tarfile
import zipfile

from pathlib import Path

from collections import namedtuple, OrderedDict

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import cv2

import requests
import base64
import json
import uuid

import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

from scipy.spatial import distance as dist

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
# What model to download.
from datetime import datetime

from model.camera import Camera
from model.camera import CameraType

import traceback

import logging

import boto3

global cameras
cameras = {}

from PIL import Image

import settings
import utils.aug_txt as aug_txt
import pytesseract
from pyzbar import pyzbar

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def close_running_threads():
    print(cameras)
    for cam in cameras.values():
        print("closing camera " + str(cam))
        cam.release()
    if settings.pgsystem is not None:
        settings.pgsystem.ReleaseInstance()
        del settings.pgsystem
        settings.pgsystem = None
    print("done closing")
atexit.register(close_running_threads)
import tensorflow as tf

class Cameras:
    def __init__(self, data):
        self.data = data
        self.params = list()
        self.sess = None

    def save_camera_config(self):
        with open('cameras.json', 'w') as outfile:
            output = {}
            for key,value in cameras.items():
                output[key]=value.__getstate__()
            json.dump(output, outfile)

    def get_cam(self,cam):
        if str(cam) not in cameras:
            config_file = Path('cameras.json')
            try:
                if config_file.is_file():
                    print(config_file)
                    with open('cameras.json', 'r') as f:
                        data = json.load(f)
                        if str(cam) in data:
                            value = data[str(cam)]
                            print("creating camera {} {}".format(cam,value['index']))
                            cameras[str(cam)] = Camera(str(cam),config=value,type=CameraType[value['type']])
            except:
                print("unable to load cameras from camera.json")
            if str(cam) not in cameras:
                try:
                    cameras[str(cam)] = Camera(cam,self.data["yres"],self.data["xres"])
                except FileNotFoundError:
                    logging.debug("instance not found {}".format(str(cam)))
                    if settings.data['SPINNAKER']:
                        try:
                            cameras[str(cam)] = Camera(cam,self.data["yres"],self.data["xres"],type=CameraType.pgrey)
                        except:
                            logging.debug(traceback.format_exc())
        if not str(cam) in cameras:
            return None
        return cameras[str(cam)]

    def remove_cam(self,cam):
        if str(cam) in cameras:
            print("removing camera " + str(cam))
            #cameras[str(cam)].set(cv2.CAP_PROP_FRAME_WIDTH,self.data["xres"]);
            #cameras[str(cam)].set(cv2.CAP_PROP_FRAME_HEIGHT,self.data["yres"]);
            cameras[str(cam)].release()
            del cameras[str(cam)]
            print("shutting down " + str(cam))

    def change_res(self,cam,x,y):
        print('setting resolution of cam %d to %dx%d',cam,x,y)
        cam = self.get_cam(cam)
        cam.setDimensions(x,y)

    def read_picture(self,file,predict=False,project=None):
            img = cv2.imread(file)
            ids = []
            detections = 0
            if predict:

                return self.predict(img,project=project)
            return img,ids,detections
            
    def predict(self,frame,cam=None,project=None):
        height, width = frame.shape[:2]
        np_image_data = np.asarray(frame)
        if self.data["bw"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ids = []
        coords = []
        detections = 0
        ret,img = cv2.imencode(self.data["image_type"],frame, self.params)
        pmp = -1
        ppmm = -1
        bimg = None
        height, width = np.array(frame).shape[:2]    
        if project.grpc:
            output_dict = self.run_inference_for_single_image_grpc(img,project)
        else:
            output_dict = self.run_inference_for_single_image(img,project)
        if "num_detections" in output_dict:
            detections = output_dict['num_detections'].tolist()
            
            if not cam == None and hasattr(cam,"waypoint"):
                (mX, mY) = (cam.waypoint[0],cam.waypoint[1])
            else:
                (mX, mY) = (0,0)

            #frame = cv2.drawMarker(frame,(int(mX),int(mY)),(0, 255, 255), cv2.MARKER_CROSS, 40, 6)


            for x in range(output_dict['num_detections']):
                box =output_dict['detection_boxes'][x].tolist()
                tbox = [height*box[0],width*box[1],height*box[2],width*box[3]]

                catt = project.labelmap_dict[output_dict['detection_classes'][x]].split('_')
                ocr = ''
                (oX, oY) = midpoint([tbox[1]-mX,tbox[0]-mY],[tbox[3]-mX,tbox[2]-mY])

                #cv2.drawMarker(frame,(int(oX),int(oY)),(0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                d = {'tag': catt[0],'flag':catt[1], 'otype':catt[2],'score': output_dict['detection_scores'][x],
                            'box': output_dict['detection_boxes'][x].tolist()}
                if not cam == None and cam.calibrated:
                    dA = dist.euclidean((oX, oY), (mX, mY))
                    mmd = dA / cam.ppmm
                    d['distance_waypoint_px'] = dA
                    d['distance_waypoint_mm']:mmd
                    d['midpoint_px']=[oX,oY]
                    d['midpoint_mm']=[oX/cam.ppmm,oY/cam.ppmm]
                if catt[2] == 'tagOCR':
                    fl = '/tmp/{}.png'.format(str(uuid.uuid4()))
                    try:
                        ci = frame[int(tbox[0]-3):int(tbox[2]+3),int(tbox[1]-3):int(tbox[3]+3)]
                        cheight, cwidth = ci.shape[:2]
                        dim = [cwidth,cheight]
                        if max(dim) < 400:
                            scalef = 400/max(dim)
                            swidth = int(ci.shape[1] * scalef)
                            sheight = int(ci.shape[0] * scalef)
                            sdim = (swidth, sheight)
                            ci = cv2.resize(ci, sdim, interpolation = cv2.INTER_CUBIC)
                        ci = ci[:,:,:3]
                        
                        cv2.imwrite(fl,ci)
                        ocr = aug_txt.detect_text(fl)
                        d['ocr'] = ocr
                    except:
                        var = traceback.format_exc()
                        d['ocr'] = ''
                        d['ocr-error']=var
                    finally:
                        if os.path.exists(fl):
                            os.remove(fl)

                elif catt[2] == 'tagBarcode':
                    try:
                        bar_ar = []
                        ci = frame[int(tbox[0]-3):int(tbox[2]+3),int(tbox[1]-3):int(tbox[3]+3)]
                        ci = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
                        if min(dim) < 300:
                            scalef = 300/min(dim)
                            swidth = int(ci.shape[1] * scalef)
                            sheight = int(ci.shape[0] * scalef)
                            sdim = (swidth, sheight)
                            ci = cv2.resize(ci, sdim, interpolation = cv2.INTER_CUBIC)
                        barcodes = pyzbar.decode(ci)
                        # loop over the detected barcodes
                        for barcode in barcodes:
                            barcodeData = barcode.data.decode("utf-8")
                            barcodeType = barcode.type
                            bc = {'data':barcodeData,'type':barcodeType}
                            bar_ar.append(bc)

                            # print the barcode type and data to the terminal
                            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
                        d['barcode']=bar_ar
                    except:
                        var = traceback.format_exc()
                        d['barcode'] = []
                        d['barcode-error']=var
                
            
                #cv2.line(frame, (int(oX), int(oY)), (int(mX), int(mY)),(255, 0, 255), 2)
                #np_image_data = np.asarray(frame)
                ids.append(d)

            bframe = vis_util.visualize_boxes_and_labels_on_image_array(
                    np_image_data,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    project.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
            ret,bimg = cv2.imencode(self.data["image_type"],bframe, self.params)
        ret,img = cv2.imencode(self.data["image_type"],frame, self.params)    
        return img,bimg,ids,detections,ppmm


    def get_picture(self,cam,predict=False,project=None,calibrate=True):
        cam = self.get_cam(cam)
        print(cam)
        ppmm = -1
        if cam.calibrated:
            ppmm = cam.ppmm
        frame,(h,w)= cam.getFrame(calibrate=calibrate)
        
        height, width = frame.shape[:2]
        if frame is None:
            print('frame not found in {}'.format(cam.location))
            return None, None,None
        np_image_data = np.asarray(frame)
        if self.data["bw"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ids = []
        coords = []
        detections = 0
        ret,img = cv2.imencode(self.data["image_type"],frame, self.params)
        if predict:
            pmp = -1
            
            if project.grpc:
                output_dict = self.run_inference_for_single_image_grpc(img,project)
            else:
                output_dict = self.run_inference_for_single_image(img,project)
            if "num_detections" in output_dict:
                detections = output_dict['num_detections'].tolist()
                
                if hasattr(cam,"waypoint"):
                    (mX, mY) = (cam.waypoint[0],cam.waypoint[1])
                else:
                    (mX, mY) = (0,0)

                #frame = cv2.drawMarker(frame,(int(mX),int(mY)),(0, 255, 255), cv2.MARKER_CROSS, 40, 6)


                for x in range(output_dict['num_detections']):
                    box =output_dict['detection_boxes'][x].tolist()
                    tbox = [height*box[0],width*box[1],height*box[2],width*box[3]]

                    catt = project.labelmap_dict[output_dict['detection_classes'][x]].split('_')
                    ocr = ''
                    (oX, oY) = midpoint([tbox[1]-mX,tbox[0]-mY],[tbox[3]-mX,tbox[2]-mY])

                    #cv2.drawMarker(frame,(int(oX),int(oY)),(0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                    d = {'tag': catt[0],'flag':catt[1], 'otype':catt[2],'score': output_dict['detection_scores'][x],
                                'box': output_dict['detection_boxes'][x].tolist()}
                    if cam.calibrated:
                        dA = dist.euclidean((oX, oY), (mX, mY))
                        mmd = dA / cam.ppmm
                        d['distance_waypoint_px'] = dA
                        d['distance_waypoint_mm']:mmd
                        d['midpoint_px']=[oX,oY]
                        d['midpoint_mm']=[oX/cam.ppmm,oY/cam.ppmm]
                    if catt[2] == 'tagOCR':
                        fl = '/tmp/{}.png'.format(str(uuid.uuid4()))
                        try:
                            ci = frame[int(tbox[0]-3):int(tbox[2]+3),int(tbox[1]-3):int(tbox[3]+3)]
                            cheight, cwidth = ci.shape[:2]
                            dim = [cwidth,cheight]
                            if max(dim) < 400:
                                scalef = 400/max(dim)
                                swidth = int(ci.shape[1] * scalef)
                                sheight = int(ci.shape[0] * scalef)
                                sdim = (swidth, sheight)
                                ci = cv2.resize(ci, sdim, interpolation = cv2.INTER_CUBIC)
                            ci = ci[:,:,:3]
                            
                            cv2.imwrite(fl,ci)
                            logging.debug('writing ' +fl)
                            ocr = aug_txt.detect_text(fl)
                            d['ocr'] = ocr
                        except:
                            var = traceback.format_exc()
                            d['ocr'] = ''
                            d['ocr-error']=var
                        finally:
                            if os.path.exists(fl):
                                os.remove(fl)

                    elif catt[2] == 'tagBarcode':
                        try:
                            bar_ar = []
                            ci = frame[int(tbox[0]-3):int(tbox[2]+3),int(tbox[1]-3):int(tbox[3]+3)]
                            ci = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
                            if min(dim) < 300:
                                scalef = 300/min(dim)
                                swidth = int(ci.shape[1] * scalef)
                                sheight = int(ci.shape[0] * scalef)
                                sdim = (swidth, sheight)
                                ci = cv2.resize(ci, sdim, interpolation = cv2.INTER_CUBIC)
                            barcodes = pyzbar.decode(ci)
                            # loop over the detected barcodes
                            for barcode in barcodes:
                                barcodeData = barcode.data.decode("utf-8")
                                barcodeType = barcode.type
                                bc = {'data':barcodeData,'type':barcodeType}
                                bar_ar.append(bc)

                                # print the barcode type and data to the terminal
                                print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
                            d['barcode']=bar_ar
                        except:
                            var = traceback.format_exc()
                            d['barcode'] = []
                            d['barcode-error']=var
                
                    #cv2.line(frame, (int(oX), int(oY)), (int(mX), int(mY)),(255, 0, 255), 2)
                    #np_image_data = np.asarray(frame)

                    
                
                    
                    ids.append(d)
            
        ret,img = cv2.imencode(self.data["image_type"],frame, self.params)
        return img,ids,detections,ppmm

    def load_image_into_numpy_array(self,img):
      return np.reshape(img,(self.data["yres"],self.data["xres"],3)).astype(np.uint8)

    def getstream(self,cam_number,predict=False,project=None):
            cam = self.get_cam(cam_number)
            frame, _ = cam.getFrame()
            
            if frame is None:
                return None
            if predict:
                height, width = frame.shape[:2]
                ret,img = cv2.imencode(".jpg",frame)
                np_image_data = np.asarray(frame)
                if project.grpc:
                    output_dict = self.run_inference_for_single_image_grpc(img,project)
                else:
                    output_dict = self.run_inference_for_single_image(img,project)
                frame = vis_util.visualize_boxes_and_labels_on_image_array(
                    np_image_data,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    project.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                if cam.calibrated:
                    if hasattr(cam,"waypoint"): 
                        (mX, mY) = (cam.waypoint[0],cam.waypoint[1])
                    else:
                        (mX, mY) = (0,0)

                    print(mY,mX)
                    cv2.drawMarker(frame,(int(mX),int(mY)),(0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                        
                for x in range(output_dict['num_detections']):
                    box =output_dict['detection_boxes'][x].tolist() 
                    tbox = [height*box[0],width*box[1],height*box[2],width*box[3]]
                    catt = project.labelmap_dict[output_dict['detection_classes'][x]].split('_')
                    ocr = ''
                    (oX, oY) = midpoint([tbox[1],tbox[0]],[tbox[3],tbox[2]])
            ret,img = cv2.imencode(".jpg",frame)
            return img

    def run_inference_for_single_image(self,image,project):
        # Encode the image in base64
        encoded_string = base64.b64encode(image).decode("utf-8")
        imgjs = {'b64':encoded_string}
        d=json.dumps({'instances':[imgjs]})
        # Json post should look like {instances:[{'b64':'encoded'}]
        url = "http://{}:{}/v1/models/{}/versions/{}:predict".format(project.tfserver,project.tfport,project.model,project.version)
        #logging.debug(d)
        #logging.debug("connecting to {}".format(url))
        r = requests.post(url, data=d)
        #logging.debug(r.status_code)
        #logging.debug(r.text)
        result = r.json()['predictions'][0]
        output_dict = {}
        output_dict['num_detections'] = np.squeeze(result['num_detections']).astype(np.int32)
        output_dict['detection_classes'] = np.squeeze(result['detection_classes']).astype(np.int32)
        output_dict['detection_boxes'] = np.reshape(result['detection_boxes'],[100,4])
        output_dict['detection_scores'] = result['detection_scores']
        return output_dict

    def run_inference_for_single_image_grpc(self,image,project):
          # Get handles to input and output tensors
          request = predict_pb2.PredictRequest()
          # Specify model name (must be the same as when the TensorFlow serving serving was started)
          request.model_spec.name = project.model
          dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
          tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
          tensor_proto = tensor_pb2.TensorProto(
            dtype=types_pb2.DT_STRING,
            tensor_shape=tensor_shape_proto,
            string_val=[image.tobytes()])
          # Initalize prediction
          # Specify signature name (should be the same as specified when exporting model)
          request.model_spec.signature_name = ''
          request.inputs['inputs'].CopyFrom(
                  tensor_proto)
          result = project.stub.Predict(request, 10.0)
          # Run inference
          output_dict = {}
          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = np.squeeze(result.outputs['num_detections'].float_val).astype(np.int32)
          output_dict['detection_classes'] = np.squeeze(result.outputs['detection_classes'].float_val).astype(np.int32)
          output_dict['detection_boxes'] = np.reshape(result.outputs['detection_boxes'].float_val,[100,4])
          output_dict['detection_scores'] = result.outputs['detection_scores'].float_val
          return output_dict
