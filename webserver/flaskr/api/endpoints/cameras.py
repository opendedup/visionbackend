import logging

from PIL import Image
import io
import base64

from flask import request
from flask_restplus import Resource, Namespace, fields

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
from flask import stream_with_context

from model.cameras import Cameras
from model.camera import Camera




from scipy.spatial import distance as dist
import math


import settings
if 'SPINNAKER' in os.environ and os.environ['SPINNAKER'].lower() =='true':
    from  utils.pgcam import list_cameras

import cv2
import imutils
from imutils import perspective
from imutils import contours
import v4l2
import fcntl

from pyzbar import pyzbar
import zbar

log = logging.getLogger(__name__)

ns = Namespace('camera', description='Camera Management')

cartesian_xy = fields.List(fields.Float,min_items=2,max_items=2,description='cartesian coordinates',example=[0,0])
cartesian_xyz = fields.List(fields.Float,min_items=3,max_items=3,description='cartesian coordinates',example=[0,0])

calibration = ns.model('Calibration', {
    'camera': fields.String(required=False, description='The camera location',
                          example='/dev/video2'),
    'image_height':fields.Integer(required=False, description='The height of the image calibrated',
                           example=1080),
    'image_width':fields.Integer(required=False, description='The width of the image calibrated',
                           example=1440),
    'checkerboard_height':fields.Integer(required=False, description='The height of the checkerboard calibrated',
                           example=5),
    'checkerboard_width':fields.Integer(required=False, description='The width of the checkerboard calibrated',
                           example=7),
    'ppmm':fields.Float(required=False, description='The pixels per MM',
                           example=9.0),
    'corners':fields.List(cartesian_xy,required=True, description='The cartesian coordinates found for the box corners.'),
    'pattern_points':fields.List(cartesian_xyz,required=True, description='The absolute pattern points calculated by the square_size.'
                             ),
    'base64': fields.String(required=False, description='The base64 encoded thumbnail of the image',
                           example="ADEW43")                         
})

resolution = ns.model('Resolution', {
    'height':fields.Integer(required=False, description='The image height',
                           example=1080),
    'width':fields.Integer(required=False, description='The image width',
                           example=1440),
})

angle = ns.model('Angle_Waypoint', {
    'degrees':fields.Float(required=False, description='The image height',
                           example=180),
    'waypoint': fields.List(fields.Float,min_items=2,max_items=2,description='cartesian coordinates for the set midpoint',example=[0,0])
})

camerans = ns.model('Camera',{
    'index' : fields.Integer(required=True, description='Camera id',
                           example=0),
    'name' : fields.String(required=True,description='The camera name as seem by the OS',
                            example='Integrated Camera'),
    'bus_info' : fields.String(required=True,description='The USB Bus id',
                            example='usb-0000:00:14.0-6'),
    'location' : fields.String(required=True,description='The location in the dev file system',
                            example='/dev/video0'),
    'h' : fields.Integer(required=True,description='Image Height',
                            example='480'),
    'w' : fields.Integer(required=True,description='Image Width',
                            example='640'),
    'calibrated' : fields.Boolean(required=True,description='Whether the camera has been calibrated',
                            example=False,default=False),
    'ppmm'  : fields.Float(required=False,description="Pixels per MM after calibration", example='8.367',default=-1),
    'waypoint':fields.List(fields.Float,min_items=2,max_items=2,description='cartesian coordinates for the set midpoint',example=[0,0]),
    'angle_offset':fields.Float(required=False, description='The adjusted angle for offset',
                           example=0)

})

def str2bool(v):
  return v.lower() in ("yes", "true","1")


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def rotation(pts):
    coords = []
    for p in pts:
        coords.append(p[0])
        coords.append(p[1])
    center = [(coords[0] + coords[4]) / 2,
                  (coords[1] + coords[5]) / 2];
    width = max([coords[4],coords[0]]) - min([coords[4],coords[0]])
    height = max([coords[3],coords[1]]) - min([coords[3],coords[1]])
    diffs = [coords[6] - coords[0], coords[7] - coords[1]]
    if diffs[1] ==0:
        diffs[1] = 1
    if diffs[0] ==0:
        diffs[0] = 1
    rotation = math.atan(diffs[0]/diffs[1]) * 180 / math.pi
    rotation = rotation -90

    if diffs[1] < 0:
        rotation += 180
    elif diffs[0] < 0:
        rotation += 360
    
    if rotation  < 0:
        rotation +=360

    return center,width,height, rotation

def subimage(image, center, theta):

   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=-theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   return image

@ns.route('/')
class GetCameras(Resource):
    @ns.marshal_list_with(camerans)
    def get(self):
        """
        Returns list of cameras attached to the system.
        """
        index = 0
        arr = []
        while True:
            
            vidfile = Path('/dev/video{}'.format(index))
            if not vidfile.is_char_device():
                break
            else:
                cam = settings.camera.get_cam(index)
                arr.append(cam.__getstate__())
            index += 1
        if settings.data['SPINNAKER']:
            nar = list_cameras()
            for r in nar:
                cam = settings.camera.get_cam(r['index'])
                arr.append(cam.__getstate__())
        
        return arr,200
@ns.route('/snap/<int:cam>')
@ns.route('/snap/<int:cam>/<string:calibrated>')
@ns.doc(params={'cam': 'The numeric id of a camera'})
@ns.doc(params={'calibrated': 'Return a calibrated image if true. (true|false)'})
class Snap(Resource):
    @ns.produces(['image/jpeg'])
    def get(self,cam,calibrated="true"):
        """
        Takes a snapshot from the camera
        """
        calb = str2bool(calibrated)
        image_name = uuid.uuid4()
        img,_,_,_ = settings.camera.get_picture(cam,predict=False,calibrate=calb)
        if img is None:
            return {'msg':"camera cannot read {}".format(0),'type':'error','target':cam}, 404
        response = make_response(img.tobytes())
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set(
            'Content-Disposition', 'attachment', filename='%s.jpg' % image_name)
        return response
@ns.route('/resolution/<int:cam>')
@ns.doc(params={'cam': 'The numeric id of a camera'})
class Resolution(Resource):
    @ns.marshal_with(resolution)
    def get(self,cam):
        """
        returns the camera resolution
        """
        cam = settings.camera.get_cam(cam)
        return {'height':cam.h,'width':cam.w}

    @ns.expect(resolution)
    def post(self,cam):
        """
        sets the resolution for a camera
        """
        data = request.json
        cam = settings.camera.get_cam(cam)
        cam.setDimensions(data['width'],data['height'])
        settings.camera.save_camera_config()
        return cam.__getstate__()

@ns.route('/angle_offset/<int:cam>')
@ns.doc(params={'cam': 'The numeric id of a camera'})
class AngleOffset(Resource):
    @ns.marshal_with(angle)
    def get(self,cam):
        """
        returns the camera resolution
        """
        cam = settings.camera.get_cam(cam)
        if not hasattr(cam,'angle_offset'):
            cam.angle_offset=0
        if not hasattr(cam,'waypoint'):
            cam.waypoint = [0,0]
        return {'degrees':cam.angle_offset,'waypoint':cam.waypoint}

    @ns.expect(angle)
    def post(self,cam):
        """
        sets the resolution for a camera
        """
        data = request.json
        cam = settings.camera.get_cam(cam)
        cam.setAngle(data['degrees'])
        cam.setWaypoint(data['waypoint'])
        settings.camera.save_camera_config()
        return cam.__getstate__()

@ns.route('/barcode/<int:cam>')
class BarCode(Resource):
    @ns.produces(['image/jpeg'])
    def get(self,cam):
        """
        Gets barcodes seen on a camera
        """
        cam = settings.camera.get_cam(cam)
        
        img,_ = cam.getFrame(calibrate=True)
        zimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(zimg)
        # loop over the detected barcodes
        for barcode in barcodes:
        	# extract the bounding box location of the barcode and draw the
        	# bounding box surrounding the barcode on the image
            print(barcode.location)
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        	# the barcode data is a bytes object so if we want to draw it on
        	# our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

        	# draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        	    0.5, (0, 0, 255), 2)

        	# print the barcode type and data to the terminal
            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        ret,measured_img = cv2.imencode('.jpg',img)
        response = make_response(measured_img.tobytes())
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set(
            'Content-Disposition', 'inline')
        return response

@ns.route('/rotation/stream/<int:cam>')
class StreamRot(Resource):
    @ns.produces(['multipart/x-mixed-replace; boundary=--jpgboundary'])
    def get(self,cam):
        """
        Stream video finding qr code for rotation and midpoint orientation
        """
        def kgen(cam):
            """Video streaming generator function."""
            while True:
                _cam = settings.camera.get_cam(cam)
        
                img,_ = _cam.getFrame(calibrate=True)
                
                zimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scanner = zbar.Scanner()
                results = scanner.scan(zimg)
                for result in results:
                    r = {'type':result.type,
                        'data':result.data.decode("ascii"),
                        'quality':result.quality,
                        'absolute-position-px':{'topleft':result.position[0],
                                    'topright':result.position[1],
                                    'bottomright':result.position[2],
                                    'bottomleft':result.position[3]}
                        }
                    for p in result.position:
                        cv2.circle(img,(p[0], p[1]), 5, (0,255,0), -1)
                    center,width,height,angle = rotation(result.position)
                    cv2.circle(img,(int(center[0]), int(center[1])), 5, (0,255,0), -1)
                    r['center'] = center
                    r['width'] = width
                    r['height'] = height
                    r['angle'] = angle
                    r['relative-position-px']= {
                        'topleft':[result.position[0][0]-center[0],result.position[0][1]-center[1]],
                        'topright':[result.position[1][0]-center[0],result.position[1][1]-center[1]],
                        'bottomright':[result.position[2][0]-center[0],result.position[2][1]-center[1]],
                        'bottomleft':[result.position[3][0]-center[0],result.position[3][1]-center[1]]}
                    img = subimage(img,(center[0],center[1]),angle)
                ret,measured_img = cv2.imencode('.jpg',img)
                
                yield (b'--jpgboundary\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(measured_img) + b'\r\n')
        resp = Response(stream_with_context(kgen(cam)),mimetype='multipart/x-mixed-replace; boundary=--jpgboundary')
        return resp


@ns.route('/rotation/<int:cam>')
class Rot(Resource):
    def get(self,cam):
        """
        Get the rotation and midpoint by scanning qr code in field of view
        """
        cam = settings.camera.get_cam(cam)
        
        img,_ = cam.getFrame(calibrate=True)
        
        zimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scanner = zbar.Scanner()
        results = scanner.scan(zimg)
        ar = []
        for result in results:
            r = {'type':result.type,
                    'data':result.data.decode("ascii"),
                    'quality':result.quality,
                    'absolute-position-px':{'topleft':result.position[0],
                        'topright':result.position[1],
                        'bottomright':result.position[2],
                        'bottomleft':result.position[3]}
                }
            center,width,height,angle = rotation(result.position)
            r['center'] = center
            r['width'] = width
            r['height'] = height
            r['angle'] = angle
            r['relative-position-px']= {
                        'topleft':[result.position[0][0]-center[0],result.position[0][1]-center[1]],
                        'topright':[result.position[1][0]-center[0],result.position[1][1]-center[1]],
                        'bottomright':[result.position[2][0]-center[0],result.position[2][1]-center[1]],
                        'bottomleft':[result.position[3][0]-center[0],result.position[3][1]-center[1]]}
            ar.append(r)
            if hasattr(cam,'ppmm'):
                r['ar'] = cam.ppmm
            else:
                r['ppmm'] = -1
        return ar


@ns.route('/measure/<int:cam>')
@ns.doc(params={'cam': 'The numeric id of a camera'})
class Measure(Resource):
    @ns.produces(['image/jpeg'])
    def get(self,cam):
        """
        Measure the size of objects in a picture
        """
        image_name = uuid.uuid4()
        cam = settings.camera.get_cam(cam)
        img,_ = cam.getFrame(calibrate=True)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = cam.ppmm
        # loop over the contours individually
        for c in cnts:
        	# if the contour is not sufficiently large, ignore it
        	if cv2.contourArea(c) < 100:
        		continue
        	# compute the rotated bounding box of the contour
        	orig = img.copy()
        	box = cv2.minAreaRect(c)
        	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        	box = np.array(box, dtype="int")

        	# order the points in the contour such that they appear
        	# in top-left, top-right, bottom-right, and bottom-left
        	# order, then draw the outline of the rotated bounding
        	# box
        	box = perspective.order_points(box)
        	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
        	for (x, y) in box:
        		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            # unpack the ordered bounding box, then compute the midpoint
        	# between the top-left and top-right coordinates, followed by
        	# the midpoint between bottom-left and bottom-right coordinates
        	(tl, tr, br, bl) = box
        	(tltrX, tltrY) = midpoint(tl, tr)
        	(blbrX, blbrY) = midpoint(bl, br)

        	# compute the midpoint between the top-left and top-right points,
        	# followed by the midpoint between the top-righ and bottom-right
        	(tlblX, tlblY) = midpoint(tl, bl)
        	(trbrX, trbrY) = midpoint(tr, br)

        	# draw the midpoints on the image
        	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        	# draw lines between the midpoints
        	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        		(255, 0, 255), 2)
        	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        		(255, 0, 255), 2)
            # compute the Euclidean distance between the midpoints
        	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        	# if the pixels per metric has not been initialized, then
        	# compute it as the ratio of pixels to supplied metric
        	# (in this case, inches)
        	if pixelsPerMetric is None:
        		pixelsPerMetric = dB / args["width"]

            # compute the size of the object
        	dimA = dA / pixelsPerMetric
        	dimB = dB / pixelsPerMetric

        	# draw the object sizes on the image
        	cv2.putText(orig, "{:.1f}mm".format(dimA),
        		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        		0.65, (255, 255, 255), 2)
        	cv2.putText(orig, "{:.1f}mm".format(dimB),
        		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        		0.65, (255, 255, 255), 2)

        ret,measured_img = cv2.imencode('.jpg',orig)
        response = make_response(measured_img.tobytes())
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set(
            'Content-Disposition', 'inline')
        return response



@ns.route('/stream/<int:cam>')
@ns.doc(params={'cam': 'The numeric id of a camera'})
class Stream(Resource):

    #@settings.api.representation('multipart/x-mixed-replace; boundary=--jpgboundary')
    @ns.produces(['multipart/x-mixed-replace; boundary=--jpgboundary'])
    def get(self,cam):
        """
        Streams the video from the designated camera.
        """
        print('calling stream for camera {}'.format(cam))
        def kgen(cam):
            """Video streaming generator function."""
            while True:
                frame = settings.camera.getstream(cam)
                
                yield (b'--jpgboundary\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
        resp = Response(stream_with_context(kgen(cam)),mimetype='multipart/x-mixed-replace; boundary=--jpgboundary')
        return resp

@ns.route('/calibrate/<int:cam>/<int:h>/<int:w>/<int:square_size>')
@ns.doc(params={'cam': 'The numeric id of a camera',
                'h':'The height of the checkerboard, in blocks, from the top left to the bottom left.',
                'w':'The width of the checkerboard, in blocks, from the top lef to the top right.',
                'square_size': 'the size of the squre in Millimeters'})
class Calibrate(Resource):
    @ns.response(204, 'Calibration successfully removed.')
    def delete(self, cam,h,w,square_size):
        """
        Deletes a project.
        """
        cam = settings.camera.get_cam(cam)
        cam.calibrated = False
        cam.setPixelsPerMM(-1)
        settings.camera.save_camera_config()
        return "Calibration successfully removed."

    @ns.response(200, 'Checkerboard foud.')
    @ns.response(404, 'Checkerboard not found.')
    @ns.marshal_with(calibration)
    def get(self,cam,h,w,square_size):
        """
        Outputs corners of a checkerboard
        """
        try:
            
            cam = settings.camera.get_cam(cam)
            img,_ = cam.getFrame(calibrate=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pattern_size = (h-1, w-1)
            pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
            pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
            pattern_points *= square_size
            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                ih, iw = img.shape[:2]
                ret,_img = cv2.imencode('.jpg',img)
                image = Image.open(io.BytesIO(_img.tobytes()))
                image.thumbnail((128, 128), Image.ANTIALIAS)
                imgByteArr = io.BytesIO()
                image.save(imgByteArr, format='JPEG')
                imgByteArr = imgByteArr.getvalue()
                ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
                return {'camera':cam.location,'image_height':ih,'image_width':iw,
                        'corners':corners.reshape(-1, 2).tolist(),'pattern_points':pattern_points.tolist(),
                        'checkerboard_height':h,'checkerboard_width':w,'ppmm':0,'base64':ncoded_string},200
            else:
                return "checkerboard not found on camera {}".format(cam.location), 404
        except Exception as inst:
            print(inst)
            return "internal error".format(cam.location), 500

    @ns.expect([calibration])
    def post(self,cam,h,w,square_size):
        """
        Outputs and saves calibration info based on an array of corners found in get method
        """
        data = request.json
        obj_points = []
        img_points = []
        for output in data:
            img_points.append(np.array(output['corners'],np.float32))
            obj_points.append(np.array(output['pattern_points'],np.float32))
        cam = settings.camera.get_cam(cam)
        img,_ = cam.getFrame(calibrate=False)
        ih, iw = img.shape[:2]
        print('h={} w={}'.format(ih,iw))
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (iw, ih), None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (iw, ih), 1, (iw, ih))
        img,_ = cam.getFrame(calibrate=False)
        cv2.imwrite('/tmp/test0.jpg',img)
        img = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        cv2.imwrite('/tmp/test1.jpg',img)
        x, y, iw, ih = roi
        img = img[y:y+ih, x:x+iw]
        cv2.imwrite('/tmp/test2.jpg',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('/tmp/test3.jpg',gray)
        pattern_size = (h-1, w-1)
        print(pattern_size)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        rot = 0
        ppmm =0
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            measurements = []
            ca = corners[0]
            for x in range(len(corners)):
                if (x+2) > len(corners):
                    break
                nxt = x+1
                if rot <(h-2):
                    D = dist.euclidean((corners[x][0][0], corners[x][0][1]), (corners[nxt][0][0], corners[nxt][0][1]))
                    measurements.append(D)
                    rot +=1
                    print(D)
                else:
                    rot =0
            ppmm = np.mean(measurements)/square_size
            cam.setPixelsPerMM(ppmm)
            cam.updateCalibration(square_size,rms,camera_matrix,dist_coefs)
            settings.camera.save_camera_config()
            return {'ppmm':ppmm,'rms':rms,'camera_matrix':camera_matrix.tolist(),
                    'dist_coefs_shape': dist_coefs.shape,'dist_coefs':dist_coefs.ravel().tolist()},201
        else:
            print('not found')
            cv2.imwrite('/tmp/test.jpg',gray)
            return "checkerboard not found on camera {}".format(cam.location), 404
