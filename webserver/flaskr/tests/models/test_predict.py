import unittest
import settings
import traceback
import logging
import sys
import json
import base64
import requests
import numpy as np
import os

from model.camera import Camera
from model.camera import CameraType

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestPredict(unittest.TestCase):
    def test_predict(self):
        pth = os.path.join(PARENT_DIR, "resources/storage_files/9389dd15-b59f-43ce-87a1-00da5dbe71a9.jpg")
        with open(pth, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        imgjs = {'b64':encoded_string}
        d=json.dumps({'instances':[imgjs]})
        print(d)
        # Json post should look like {instances:[{'b64':'encoded'}]
        url = "http://{}:{}/v1/models/gummies10:predict".format("192.168.0.103",8080)
        #logging.debug(d)
        logger.debug("connecting to {}".format(url))
        r = requests.post(url, json=d)
        logger.debug(r.status_code)
        logger.debug(r.text)
        result = r.json()['predictions'][0]
        output_dict = {}
        output_dict['num_detections'] = np.squeeze(result['num_detections']).astype(np.int32)
        output_dict['detection_classes'] = np.squeeze(result['detection_classes']).astype(np.int32)
        output_dict['detection_boxes'] = np.reshape(result['detection_boxes'],[100,4])
        output_dict['detection_scores'] = result['detection_scores']


if __name__ == '__main__':
    unittest.main()