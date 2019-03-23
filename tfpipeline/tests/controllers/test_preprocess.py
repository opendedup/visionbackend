import unittest
import docker
import models.job
import tests.resources.prep_job as prep_job
from controllers.preprocess import process_json, create_label_pbtxt, create_tf_example, delete_staged, upload_metadata
from utils.storage import Storage
from models.job import Job
import logging
import sys
import os

import uuid

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

client = docker.from_env()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        env = {
            "MINIO_SECRET_KEY": "imagerie",
            "MINIO_ACCESS_KEY": "imagerie"
        }
        self.container = client.containers.run('minio/minio',
                                               name='preptest', command='server /data', remove=True,
                                               environment=env, detach=True, ports={'9000/tcp': 9002})
        logger.debug(self.container)
        self.s3_storage = Storage(prep_job.s3_storage)
        self.gcs_storage = Storage(prep_job.gcs_storage)
        pth = os.path.join(PARENT_DIR, "resources/gummies2/")
        # self.gcs_storage.upload_path(pth,'{}'.format('projects'))
        self.s3_storage.upload_path(pth, '{}'.format('projects'))

    def tearDown(self):
        self.container.stop()

    def test_init_storage(self):
        for jb in prep_job.jobs:
            job = Job(jb['name'], jb)
            job.type = 'preprocess'
            job.init_temp(str(uuid.uuid4()))
            try:
                job.init_labels()
                job.init_storage()
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
            finally:
                job.cleanup()

    def test_process_json(self):
        for jb in prep_job.jobs:
            job = Job(jb['name'], jb)
            job.type = 'preprocess'
            job.init_temp(str(uuid.uuid4()))
            try:
                job.init_labels()
                job.init_storage()
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
                process_json(job)
                delete_staged(job)
            finally:
                job.cleanup()

    def test_process_all(self):
        for jb in prep_job.jobs:
            job = Job(jb['name'], jb)
            job.type = 'preprocess'
            job.init_temp(str(uuid.uuid4()))
            try:
                job.init_labels()
                job.init_storage()
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
                process_json(job)
                create_label_pbtxt(job)
                create_tf_example(job)
                create_tf_example(job, False)
                delete_staged(job)
                upload_metadata(job)
            finally:
                job.cleanup()


if __name__ == '__main__':
    unittest.main()
