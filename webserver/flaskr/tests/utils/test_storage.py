import unittest
import logging
import sys
import traceback
import tests.resources.config_storage as cfg
from utils.storage import Storage
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
class TestStorage(unittest.TestCase):
    def cleanup(self,s):
        for f in cfg.files:
            dst = "{}/{}".format(cfg.bucket_dir,f)
            s.delete_cloud_file(dst)

    def test_list(self):
        s = Storage(cfg.config)
        self._test_list(s)
        s = Storage(cfg.gsconfig)
        self._test_list(s)
    
    def test_list_folders(self):
        s = Storage(cfg.config)
        self._test_list_folders(s)
        s = Storage(cfg.gsconfig)
        self._test_list_folders(s)
    
    def test_list_metadata(self):
        s = Storage(cfg.config)
        self._test_list_metadata(s)
        s = Storage(cfg.gsconfig)
        self._test_list_metadata(s)
    
    def test_write_string(self):
        s = Storage(cfg.config)
        self._test_write_string(s)
        s = Storage(cfg.gsconfig)
        self._test_write_string(s)

    def _test_list(self,s):
        
        try:
            
            for f in cfg.files:
                pth = os.path.join(PARENT_DIR, "resources/storage_files/"+f)
                dst = "{}/{}".format(cfg.bucket_dir,f)
                logging.debug("uploading to " +dst)
                s.upload_file(pth,dst)
            lst = s.list_files(cfg.bucket_dir)
            nlst = []
            for l in lst:
                nlst.append(l[len(cfg.bucket_dir)+1:])
            self.assertEqual(set(nlst),set(cfg.files))
        finally:
            self.cleanup(s)
    
    def _test_list_folders(self,s):
        try:
            
            for f in cfg.files:
                pth = os.path.join(PARENT_DIR, "resources/storage_files/"+f)
                dst = "{}/{}".format(cfg.bucket_dir,f)
                logging.debug("uploading to " +dst)
                s.upload_file(pth,dst)
            lst = s.list_files(cfg.bucket_dir,delimiter="/",folders_only=True)
            nlst = []
            for l in lst:
                nlst.append(l[len(cfg.bucket_dir)+1:])
            print('***************')
            print(cfg.bucket_dir)
            print(nlst)
            print("!!!!!!!!!!!")
        finally:
            self.cleanup(s)
    
    def _test_list_metadata(self,s):
        try:
            for f in cfg.files:
                pth = os.path.join(PARENT_DIR, "resources/storage_files/"+f)
                dst = "{}/{}".format(cfg.bucket_dir,f)
                s.upload_file(pth,dst)
            lst = s.list_files(cfg.bucket_dir,metadata=True)
            nlst = []
            for l in lst:
                self.assertIsNotNone(l['name'])
                self.assertIsNotNone(l['ETag'])
                self.assertIsNotNone(l['lastModified'])
                self.assertIsNotNone(l['size'])
                nlst.append(l['name'][len(cfg.bucket_dir)+1:])
            self.assertEqual(set(nlst),set(cfg.files))
        finally:
            self.cleanup(s)

    def _test_write_string(self,s):
        fl = "{}/{}".format(cfg.bucket_dir,'test.txt')
        try:
            s.upload_data(cfg.test_string,fl,contentType="text/plain")
            ns = s.download_to_string(fl)
            self.assertEqual(cfg.test_string,ns)
            
        finally:
            s.delete_cloud_file(fl)

             
if __name__ == '__main__':
    unittest.main()

        


