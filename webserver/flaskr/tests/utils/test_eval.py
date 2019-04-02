import unittest
import logging
import sys
import traceback
import tests.resources.config_eval as cfg
from utils.storage import Storage
import utils.tflogs as tflogs
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
class TestEval(unittest.TestCase):

    def test_eval(self):
        s = Storage(cfg.config)
        evts = tflogs.get_events(cfg.folder,'/tmp',s)
        print(len(evts))
        print(evts)
    
    def test_eval_image(self):
        s = Storage(cfg.config)
        evts = tflogs.get_events(cfg.folder,'/tmp',s,full=True)
        print(len(evts))
        print(evts)
        
    
    

             
if __name__ == '__main__':
    unittest.main()

        


