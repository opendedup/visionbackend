import unittest
import tests.resources.config_camera as cfg
import settings
import traceback
import logging
import sys
import PySpin
import json

from model.camera import Camera
from model.camera import CameraType

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestCamera(unittest.TestCase):
    def test_init_usb_am(self):
        cam = Camera(cfg.usbcamera_id,cfg.camera_yres,cfg.camera_xres)
        self.assertIsNotNone(cam.type, CameraType.usb)
        cam.release()
    
    def test_init_pgcam(self):
        settings.pgsystem = PySpin.System.GetInstance()
        cam = Camera(cfg.pgcamera_id,cfg.camera_yres,cfg.camera_xres,type=CameraType.pgrey)
        self.assertIsNotNone(cam.type, CameraType.pgrey)
        cam.release()
        if settings.pgsystem is not None:
            settings.pgsystem.ReleaseInstance()
            del settings.pgsystem
            settings.pgsystem = None

    def test_change_res(self):
        for type in CameraType:
            if type == CameraType.pgrey:
                index = cfg.pgcamera_id
            else:
                index = cfg.usbcamera_id
            settings.pgsystem = PySpin.System.GetInstance()
            cam = Camera(index,cfg.camera_yres,cfg.camera_xres,type=type)
            cam.setDimensions(cfg.camera_nxres,cfg.camera_nyres)
            self.assertEqual(cam.h,cfg.camera_nyres)
            self.assertEqual(cam.w,cfg.camera_nxres)
            cam.release()
            if settings.pgsystem is not None:
                settings.pgsystem.ReleaseInstance()
                del settings.pgsystem
                settings.pgsystem = None
    
    def test_read(self):
        for type in CameraType:
            if type == CameraType.pgrey:
                index = cfg.pgcamera_id
            else:
                index = cfg.usbcamera_id
            settings.pgsystem = PySpin.System.GetInstance()
            cam = Camera(index,cfg.camera_yres,cfg.camera_xres,type=type)
            frame,_ = cam.getFrame()
            height, width = frame.shape[:2]
            self.assertEqual(height,cfg.camera_yres)
            self.assertEqual(width,cfg.camera_xres)
            cam.release()
            if settings.pgsystem is not None:
                settings.pgsystem.ReleaseInstance()
                del settings.pgsystem
                settings.pgsystem = None
    
    def test_tojson(self):
        for type in CameraType:
            if type == CameraType.pgrey:
                index = cfg.pgcamera_id
            else:
                index = cfg.usbcamera_id
            settings.pgsystem = PySpin.System.GetInstance()
            cam = Camera(index,cfg.camera_yres,cfg.camera_xres,type=type)
            frame,_ = cam.getFrame()
            height, width = frame.shape[:2]
            self.assertEqual(height,cfg.camera_yres)
            self.assertEqual(width,cfg.camera_xres)
            json.dump(cam.__getstate__(), '/tmp/camera.json')
            cam.release()
            if settings.pgsystem is not None:
                settings.pgsystem.ReleaseInstance()
                del settings.pgsystem
                settings.pgsystem = None


if __name__ == '__main__':
    unittest.main()

