import unittest
from utils.pgcam import list_cameras
from utils.pgcam import get_camera
from utils.pgcam import get_caminfo
from utils.pgcam import init_cam
from utils.pgcam import release_cam
from utils.pgcam import PGCam
from utils.pgcam import cam_node_cmd
import tests.resources.config_pgrey as cfg
import settings
import PySpin
import logging
import sys
import traceback
from tests.resources._AssertNotRaisesContext import _AssertNotRaisesContext
import cv2

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestPGCam(unittest.TestCase):
    def assertNotRaises(self, expected_exception, *args, **kwargs):
        context = _AssertNotRaisesContext(expected_exception, self)
        try:
            return context.handle('assertNotRaises', args, kwargs)
        finally:
            context = None

    def test_list(self):
        settings.pgsystem = PySpin.System.GetInstance()
        ar = list_cameras()
        self.assertIsNotNone(ar)
        if settings.pgsystem is not None:
            settings.pgsystem.ReleaseInstance()
            del settings.pgsystem
            settings.pgsystem = None
    
    def test_cam_attr(self):
        settings.pgsystem = PySpin.System.GetInstance()
        try:
            cam = get_camera(cfg.camera_id)
            self.assertIsNotNone(cam)
            self.assertEqual(cfg.camera_id,int(cam_node_cmd(cam,
                                        'TLDevice.DeviceSerialNumber',
                                        'GetValue',
                                        'RO')))
            cam.DeInit()
            del cam 
        except:
            self.fail(traceback.format_exc())
        
        if settings.pgsystem is not None:
            settings.pgsystem.ReleaseInstance()
            del settings.pgsystem
            settings.pgsystem = None
    
    def test_get_camera(self):
        settings.pgsystem = PySpin.System.GetInstance()
        try:
            cam = get_camera(cfg.camera_id)
            self.assertIsNotNone(cam)
            
        except:
            self.fail(traceback.format_exc())
        cam.DeInit()
        del cam 
        if settings.pgsystem is not None:
            settings.pgsystem.ReleaseInstance()
            del settings.pgsystem
            settings.pgsystem = None
    
    def test_init_cam(self):
        settings.pgsystem = PySpin.System.GetInstance()
        try:
            cam = get_camera(cfg.camera_id)
            self.assertIsNotNone(cam)
            init_cam(cam)
            image_result = cam.GetNextImage()
            self.assertIsNotNone(image_result)
            image_result.Release()
            cam.EndAcquisition()
            cam.DeInit()
            del cam 
        except:
            self.fail(traceback.format_exc())
        
        if settings.pgsystem is not None:
            settings.pgsystem.ReleaseInstance()
            del settings.pgsystem
            settings.pgsystem = None
    
    def test_get_caminfo(self):
        settings.pgsystem = PySpin.System.GetInstance()
        info = get_caminfo(cfg.camera_id)
        self.assertIsNotNone(info)

        if settings.pgsystem is not None:
            settings.pgsystem.ReleaseInstance()
            del settings.pgsystem
            settings.pgsystem = None
    
    def test_pgcam(self):
        settings.pgsystem = PySpin.System.GetInstance()
        pgcam = None
        try:
            pgcam = PGCam(cfg.camera_id,cfg.camera_yres,cfg.camera_xres)
            pgcam.release()
        except:
            self.fail(traceback.format_exc())
    
    def test_pgcam_resolution(self):
        settings.pgsystem = PySpin.System.GetInstance()
        pgcam = None
        try:
            pgcam = PGCam(cfg.camera_id,cfg.camera_yres,cfg.camera_nxres)
            pgcam.set_resolution(cfg.camera_nyres,cfg.camera_nxres)
            pgcam.release()
        except:
            self.fail(traceback.format_exc())
    
    def test_pgcam_read(self):
        settings.pgsystem = PySpin.System.GetInstance()
        pgcam = None
        try:
            pgcam = PGCam(cfg.camera_id,cfg.camera_yres,cfg.camera_xres)
            _,frame = pgcam.read()
            height, width = frame.shape[:2]
            self.assertEqual(height,cfg.camera_yres)
            self.assertEqual(width,cfg.camera_xres)
            pgcam.release()
        except:
            self.fail(traceback.format_exc())
    
    def test_pgcam_read_changeres_read(self):
        settings.pgsystem = PySpin.System.GetInstance()
        pgcam = None
        try:
            pgcam = PGCam(cfg.camera_id,cfg.camera_yres,cfg.camera_xres)
            _,frame = pgcam.read()
            height, width = frame.shape[:2]
            self.assertEqual(height,cfg.camera_yres)
            self.assertEqual(width,cfg.camera_xres)
            pgcam.set_resolution(cfg.camera_nyres,cfg.camera_nxres)
            _,frame = pgcam.read()
            height, width = frame.shape[:2]
            self.assertEqual(height,cfg.camera_nyres)
            self.assertEqual(width,cfg.camera_nxres)
            pgcam.release()
        except:
            self.fail(traceback.format_exc())



if __name__ == '__main__':
    unittest.main()


