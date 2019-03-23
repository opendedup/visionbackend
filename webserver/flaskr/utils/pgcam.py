import PySpin
import settings
import cv2
import numpy as np
import logging
import sys

import traceback
import threading


class PGCam(object):
    def __init__(self,index,h,w):
        self.index = index
        self.streaming = False
        self.lock = threading.Lock()
        self.cam = get_camera(index)
        self.set_resolution(h,w)
    
    def set_resolution(self,h,w):
        with self.lock:
            if self.streaming:
                self.cam.EndAcquisition()
                self.streaming = False
            self.h = h
            self.w = w
            if self.cam.Width.GetAccessMode() == PySpin.RW and self.cam.Width.GetInc() != 0 and self.cam.Width.GetMax != 0:
                self.cam.Width.SetValue(w)
                logging.info("Width set to %i..." % self.cam.Width.GetValue())

            else:
                logging.info("Width not available...")

            # Set maximum height
            #
            # *** NOTES ***
            # A maximum is retrieved with the method GetMax(). A node's minimum and
            # maximum should always be a multiple of its increment.
            if self.cam.Height.GetAccessMode() == PySpin.RW and self.cam.Height.GetInc() != 0 and self.cam.Height.GetMax != 0:
                self.cam.Height.SetValue(h)
                logging.info("Height set to %i..." % self.cam.Height.GetValue())

            else:
                logging.info("Height not available...")

    
    def read(self):
        with self.lock:
            if not self.streaming:
                init_cam(self.cam)
                self.streaming = True
            image_result =  None
            for _ in range(10):
                image_result = self.cam.GetNextImage()
                if image_result is None:
                    logging.debug('image is null')

                if image_result.IsIncomplete():
                    logging.info('Image incomplete with image status %d ... \n' % image_result.GetImageStatus())
                    try:
                        image_result.release()
                    except:
                        pass
                    del image_result
                    
                else:
                    break
            rawFrame = np.array(image_result.GetData(), dtype="uint8").reshape( (image_result.GetHeight(), image_result.GetWidth()) )
            image_result.Release()
            del image_result
            frame = cv2.cvtColor(rawFrame, cv2.COLOR_BAYER_BG2BGR)
            return None,frame
    
    def release(self):
        with self.lock:
            if self.streaming:
                self.cam.EndAcquisition()
            elif self.cam.IsStreaming():
                self.cam.EndAcquisition()
            if self.cam.IsInitialized():
                self.cam.DeInit()
            del self.cam
            self.streaming = False

    
    def isOpened(self):
        if self.streaming:
            return True
        if self.cam.IsStreaming():
            return True
        return False 



def print_device_info(nodemap):
    try:
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
        dev = {}

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                if node_feature.GetName() == 'DeviceSerialNumber':
                    dev['index'] = int(node_feature.ToString())
                    if 'name' in dev:
                        dev['name'] = dev['name'] + ' - ' + str(dev['index'])
                    dev['bus_info'] = 'pgrey'
                    dev['location'] = 'pgrey'
                if node_feature.GetName() == 'DeviceModelName':
                    if 'index' in dev:
                        dev['name'] = node_feature.ToString() + ' - ' + str(dev['index'])
                    else:
                        dev['name'] = node_feature.ToString()
                

        else:
            logging.warning('Device control information not available.')
            return None
        #Todo: discover dimentions appropriately
        dev['max_width'] = 2048
        dev['max_height'] = 1536
    
            
        

    except PySpin.SpinnakerException as ex:
        logging.error('Error: %s' % ex)
        return None

    return dev

def release_cam(cam):
    if cam is not None:
        logging.info('releasing camera')
        try:
            cam.EndAcquisition()
        except PySpin.SpinnakerException as ex:
            logging.debug('Error: %s' % ex)
        try:
            cam.DeInit()
        except PySpin.SpinnakerException as ex:
            logging.debug('Error: %s' % ex)
        del cam 

def init_cam(cam):
    if not cam.IsInitialized():
        logging.info('Initializing Camera')
        cam.Init()
    try:
        cam_node_cmd(cam,'TLStream.StreamBufferHandlingMode',
                                        'SetValue',
                                        'RW',
                                        'PySpin.StreamBufferHandlingMode_NewestOnly')
        nodemap = cam.GetNodeMap()
        fr_acquisition_mode = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if not PySpin.IsAvailable(fr_acquisition_mode) or not PySpin.IsWritable(fr_acquisition_mode):
            print('Unable to set framerate mode to True.')
        fr_acquisition_mode.SetValue(True)

        fr_acquisition_mode = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(fr_acquisition_mode) or not PySpin.IsWritable(fr_acquisition_mode):
            print('Unable to set framerate to 30.0.')
        fr_acquisition_mode.SetValue(30.0)
        cam_node_cmd(cam,'AcquisitionMode',
                                        'SetValue',
                                        'RW',
                                        'PySpin.AcquisitionMode_Continuous')
        cam.BeginAcquisition()
    except:
        logging.error(traceback.format_exc())

    logging.info('Camera acquisition mode set to continuous...')

    # Begin acquiring images
    

    logging.info('Camera started acquiring images...')
    for i in range(30):
        image_result = cam.GetNextImage()
        image_result.Release()
    logging.info("read {} images".format(i))

def cam_node_cmd(cam, cam_attr_str, cam_method_str, pyspin_mode_str=None, cam_method_arg=None):
    """ Performs cam_method on input cam and attribute with optional access mode check  """

    # First, get camera attribute
    cam_attr = cam
    cam_attr_str_split = cam_attr_str.split('.')
    for sub_cam_attr_str in cam_attr_str_split:
        cam_attr = getattr(cam_attr, sub_cam_attr_str)

    # Print command info
    info_str = 'Executing: "' + '.'.join([cam_attr_str, cam_method_str]) + '('
    if cam_method_arg is not None:
        info_str += str(cam_method_arg)
    print(info_str + ')"')

    # Perform optional access mode check
    if pyspin_mode_str is not None:
        if cam_attr.GetAccessMode() != getattr(PySpin, pyspin_mode_str):
            raise RuntimeError('Access mode check failed for: "' + cam_attr_str + '" with mode: "' +
                               pyspin_mode_str + '".')

    # Format command argument in case it's a string containing a PySpin attribute
    if isinstance(cam_method_arg, str):
        cam_method_arg_split = cam_method_arg.split('.')
        if cam_method_arg_split[0] == 'PySpin':
            if len(cam_method_arg_split) == 2:
                cam_method_arg = getattr(PySpin, cam_method_arg_split[1])
            else:
                raise RuntimeError('Arguments containing nested PySpin arguments are currently not '
                                   'supported...')

    # Perform command
    if cam_method_arg is None: #pylint: disable=no-else-return
        return getattr(cam_attr, cam_method_str)()
    else:
        return getattr(cam_attr, cam_method_str)(cam_method_arg)
    

def get_camera(index):
    cam_list = settings.pgsystem.GetCameras()
    rcam = None
    logging.info('getting camera {}'.format(index))
    for _, cam in enumerate(cam_list):
        node_device_serial_number = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber'))
        device_serial_number = node_device_serial_number.GetValue()
        if device_serial_number == str(index):
            rcam = cam
            break
    cam_list.Clear()
    rcam.Init()
    return rcam

def get_caminfo(index):
    ar = None
    try:
        cam_list = settings.pgsystem.GetCameras()
        

        for _, cam in enumerate(cam_list):
            # Retrieve TL device nodemap
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            device_serial_number = node_device_serial_number.GetValue()
            if device_serial_number == str(index):
                ar = print_device_info(nodemap_tldevice)
                break
        
        cam_list.Clear()

    except PySpin.SpinnakerException as ex:
        if cam_list is not None:
            cam_list.Clear()
        logging.error('Error: %s' % ex)
        ar = None

    return ar        
        

def list_cameras():
    cam_list = None
    try:
        cam_list = settings.pgsystem.GetCameras()
        ar = []

        for _, cam in enumerate(cam_list):

            # Retrieve TL device nodemap
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Print device information
            ar.append(print_device_info(nodemap_tldevice))
        cam_list.Clear()

    except PySpin.SpinnakerException as ex:
        if cam_list is not None:
            cam_list.Clear()
        logging.error('Error: %s' % ex)
        return None

    return ar