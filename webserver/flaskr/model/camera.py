import cv2
import v4l2
import fcntl
import json
import numpy as np
import traceback
import sys
from enum import Enum
import logging
import os
import imutils

if 'SPINNAKER' in os.environ and os.environ['SPINNAKER'].lower() =='true':
    from utils.pgcam import get_caminfo
    from utils.pgcam import PGCam

class CameraType(Enum):
    usb = 'usb'
    pgrey = 'pgrey'

class Camera:
    def __init__(self, index, h=480, w=640, config=None,type=CameraType.usb):
        self.type = type
        self.cap = None
        if self.type == CameraType.usb:
            self.init_usb(index,h,w,config)
        elif self.type == CameraType.pgrey:
            self.init_pg(index,h,w,config)

    def updateCalibration(self, square_size, rms, camera_matrix, dist_coefs,waypoint=[0,0],angle_offset=0):
        self.square_size = square_size
        self.rms = rms
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs

        print(self.camera_matrix)
        print(self.dist_coefs)

        self.calibration_config = {'square_size': square_size, 'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs_shape': dist_coefs.shape, 'dist_coefs': dist_coefs.ravel().tolist(),'waypoint':waypoint,'angle_offset':angle_offset}
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coefs, (self.w, self.h), 1, (self.w, self.h))
        print(self.roi)
        self.calibrated = True

    def init_pg(self,index,h,w,config):
        """Inititialize the PointGrey camera
        """
        self.index = index
        info = get_caminfo(self.index)
        logging.info(info)
        self.name = info['name']
        self.location = info['location']
        self.bus_info = info['bus_info']
        self.max_width = info['max_width']
        self.max_height = info['max_height']
        
        
        if config != None:
            
            logging.debug(config)
            if 'waypoint' in config:
                self.waypoint = config['waypoint']
                
            if 'angle_offset' in config:
                self.angle_offset = config['angle_offset']
            self.h = config['h']
            self.w = config['w']
            self.calibrated = False
            if 'calibration_config' in config:
                calibration = config['calibration_config']
                self.updateCalibration(calibration['square_size'], calibration['rms'],
                                        np.array(
                                            calibration['camera_matrix'], np.float32),
                                        np.array(calibration['dist_coefs'], np.float32).reshape(calibration['dist_coefs_shape']))

                self.square_size = calibration['square_size']
                self.rms = calibration['rms']
                self.angle = info['angle']
                self.waypoint = info['waypoint']
                self.angle_offset=info['angle_offset']
                self.camera_matrix = np.array(
                    calibration['camera_matrix'], np.float32)
                if 'ppmm' in config:
                    self.ppmm = config['ppmm']
        else:
            self.index = index
            self.h = h
            self.w = w
            self.calibrated = False
            
        


    
    def init_usb(self,index,h,w,config):
        self.angle_offset = 0
        self.waypoint=[0,0]
        if config != None:
            self.index = config['index']
            vd = open('/dev/video{}'.format(self.index), 'r')
            cp = v4l2.v4l2_capability()

            fcntl.ioctl(vd, v4l2.VIDIOC_QUERYCAP, cp)
            if config['name'] == cp.card.decode('utf-8') and cp.bus_info.decode('utf-8') == config['bus_info']:

                self.name = cp.card.decode('utf-8')
                self.bus_info = cp.bus_info.decode('utf-8')
                self.location = '/dev/video{}'.format(index)
                self.h = config['h']
                self.w = config['w']
                #Todo: Fix to discover max dimentions
                self.max_width = 3840
                self.max_height = 2160
                self.calibrated = False
                if 'waypoint' in config:
                    self.waypoint = config['waypoint']
                
                if 'angle_offset' in config:
                    self.angle_offset = config['angle_offset']

                if 'calibration_config' in config:
                    calibration = config['calibration_config']
                    self.updateCalibration(calibration['square_size'], calibration['rms'],
                                           np.array(
                                               calibration['camera_matrix'], np.float32),
                                           np.array(calibration['dist_coefs'], np.float32).reshape(calibration['dist_coefs_shape']))

                    self.square_size = calibration['square_size']
                    self.rms = calibration['rms']
                    
                    self.camera_matrix = np.array(
                        calibration['camera_matrix'], np.float32)
                    if 'ppmm' in config:
                        self.ppmm = config['ppmm']

            else:
                raise ValueError('The read config {} - {} does not match {} - {} '
                                 .format(config['name'], config['bus_info'], cp.card.decode('utf-8'), cp.bus_info.decode('utf-8')))
        else:
            self.index = index
            vd = open('/dev/video{}'.format(index), 'r')
            cp = v4l2.v4l2_capability()

            fcntl.ioctl(vd, v4l2.VIDIOC_QUERYCAP, cp)
            self.name = cp.card.decode('utf-8')
            self.bus_info = cp.bus_info.decode('utf-8')
            self.location = '/dev/video{}'.format(index)
            self.max_width = 3840
            self.max_height = 2160
            self.h = h
            self.w = w

            self.calibrated = False
    
    

    def setPixelsPerMM(self, ppmm):
        self.ppmm = ppmm

    def setDimensions(self, w, h):       
        self.h = h
        self.w = w
        if self.type == CameraType.pgrey:
            if self.cap is None:
                self.cap = PGCam(self.index,self.h,self.w)
            self.cap.set_resolution(h,w)
        else:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self.create_cap()
            if self.calibrated:
                self.calibrated = False
            for x in range(30):
                self.cap.read()

    def create_cap(self):
        if self.type == CameraType.pgrey:
            if self.cap is None:
                self.cap = PGCam(self.index,self.h,self.w)

        elif self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            for x in range(30):
                try:
                    self.cap.read()
                except:
                    print("er   ror creating capture device")
                    traceback.print_exc()
    
    def setAngle(self,angle_offset ):
        self.angle_offset=angle_offset
    
    def setWaypoint(self,waypoint):
        self.waypoint=waypoint
                    
    def getFrame(self, calibrate=True):
        try:
            self.create_cap()
            _, frame = self.cap.read()
            height, width = frame.shape[:2]
            if calibrate and self.calibrated:
                frame = cv2.undistort(
                    frame, self.camera_matrix, self.dist_coefs, None, self.newcameramtx)
                x, y, w, h = self.roi
                frame = frame[y:y+h, x:x+w]
                height, width = frame.shape[:2]
                wd = 0
                hd = 0
                if w < self.w:
                    wd = self.w - w
                if h < self.h:
                    hd = self.h - h
                if hd > 0 or wd > 0:
                    frame = cv2.copyMakeBorder(
                        frame, 0, hd, 0, wd, cv2.BORDER_CONSTANT)
            if hasattr(self,"angle_offset") and self.angle_offset !=0: 
                frame = self.subimage(frame)
            return frame,(height,width)
        except Exception as err:
            traceback.print_tb(err.__traceback__)
            print(err)
            print("error while reading {}".format(self.location))
            try:
                self.cap.release()
            except:
                print("could not release camera")
            self.create_cap()
            return None,None
    
    def subimage(self,image):
        ''' 
        Rotates OpenCV image around center with angle theta (in deg)
        then crops the image according to width and height.
        '''

        # Uncomment for theta in radians
        #theta *= 180/np.pi

        shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

        matrix = cv2.getRotationMatrix2D( center=(self.waypoint[0],self.waypoint[1]), angle=-self.angle_offset, scale=1 )
        image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

        return image
            

    def release(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.calibrated:
            self.calibration = {'square_size': self.square_size, 'rms': self.rms, 'camera_matrix': self.camera_matrix.tolist(
            ), 'dist_coefs_shape': self.dist_coefs.shape, 'dist_coefs': self.dist_coefs.ravel().tolist()}
        if 'rms' in state:
            del state['rms']
        if 'square_size' in state:
            del state['square_size']
        if 'camera_matrix' in state:
            del state['camera_matrix']
        if 'dist_coefs' in state:
            del state['dist_coefs']
        if 'newcameramtx' in state:
            del state['newcameramtx']
        if 'roi' in state:
            del state['roi']
        if 'cap' in state:
            del state['cap']
        if 'type' in state:
            state['type'] = state['type'].name
        if 'lock' in state:
            del state['lock']
        if hasattr(self, 'ppmm'):
            print(self.ppmm)
        return state
