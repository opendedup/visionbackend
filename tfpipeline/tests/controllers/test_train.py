import unittest
import models.job
import tests.resources.train_job as train_job
from controllers.train import updateFileML,upload_model,upload_packages,start_ml_engine,upload_metadata
from utils.storage import Storage
from models.job import Job
import logging
import sys
import os
import json


import uuid


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.gcs_storage = Storage(train_job.gcs_storage)
        pth = os.path.join(PARENT_DIR, "resources/gummies/traindata")
        self.gcs_storage.upload_path(pth,'{}/{}'.format('corpus','traindata2'))
        pass
    
    def tearDown(self):
        pass
    
    def test_train_mlengine(self):
        train_job.train_job['name'] = str(uuid.uuid4())
        job = Job(train_job.train_job['name'],train_job.train_job)
        job.type='preprocess'
        job.init_temp(str(uuid.uuid4()))
        try:
            logging.info("step1")
            job.init_storage()
            logging.info("step2")
            if not hasattr(job, 'label_file') or job.label_file is None:
                job.label_file='corpus/' +job.prep_name + "/object-detection.pbtxt"
            job.init_labels()
            self.assertGreater(len(job.categories),0)
            logging.info("step3")
            source = json.loads(job.download_to_string('corpus/' +job.prep_name + "/job_def.json"))
            job.project_name = source['project_name']
            logging.info("step4")
            
            updateFileML(job)
            logging.info("step5")
            upload_model(job)
            logging.info("step6")
            upload_packages(job)
            logging.info("step7")
            start_ml_engine(job)
            logging.info("step8")
            history = json.loads(job.download_to_string('corpus/' +job.prep_name + "/job_history.json",))
            upload_metadata(job,"training_jobs/" + job.name,history)
        finally:
            job.cleanup()

    def test_train_mlengine_copy(self):
        train_job.train_job['name'] = str(uuid.uuid4())
        job = Job(train_job.train_job['name'],train_job.train_job)
        job.type='preprocess'
        job.init_temp(str(uuid.uuid4()))
        try:
            logging.info("step1")
            job.init_storage()
            logging.info("step2")
            if hasattr(job,'source_training') and job.source_training is not '':
                sjd = json.loads(job.download_to_string('training_jobs/' +job.source_training + "/job_def.json"))
                job.num_train_steps += sjd['num_train_steps']
                job.model =  sjd['model']
                st = 'training_jobs/{}/'.format(job.source_training)
                dt = 'training_jobs/{}/'.format(job.name)
                job.copy_folder(st,dt)
                job.delete_cloud_file('{}{}'.format(dt,"job_def.json"))
                job.delete_cloud_file('{}{}'.format(dt,"job_history.json"))
            logging.info("step3")
            if not hasattr(job, 'label_file') or job.label_file is None:
                job.label_file='corpus/' +job.prep_name + "/object-detection.pbtxt"
            job.init_labels()
            self.assertGreater(len(job.categories),0)
            logging.info("step4")
            source = json.loads(job.download_to_string('corpus/' +job.prep_name + "/job_def.json"))
            job.project_name = source['project_name']
            logging.info("step5")
            
            updateFileML(job)
            logging.info("step6")
            upload_model(job)
            logging.info("step7")
            upload_packages(job)
            logging.info("step8")
            start_ml_engine(job)
            logging.info("step9")
            history = json.loads(job.download_to_string('corpus/' +job.prep_name + "/job_history.json",))
            upload_metadata(job,"training_jobs/" + job.name,history)
        finally:
            job.cleanup()


if __name__ == '__main__':
    unittest.main()



    