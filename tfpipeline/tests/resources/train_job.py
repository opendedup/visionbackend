prepjob ={
  "use_gcs": True,
  "aug_rules": {
    "remove_out_of_image": "cut_out_of_image",
    "rotate":[170,-90]
  },
  "aug": True,
  "train_samples": 100,
  "test_samples": 20,
  "bucket": "unittestimagerie0",
  "project_name": "gummies",
  "process_json":True,
  "name": "traindata"
}

train_corpus='gummies/traindata'

train_job = {
  'name':'training_3',
  'bucket':'unittestimagerie0',
  'model':'faster_rcnn_resnet101_coco',
  'num_train_steps':100,
  'prep_name':'data3',
  'max_dim':1920,
  'min_dim':1200,
  'batch_size':1,
  'ml_engine':True,
  'ml_workers':16,
  'use_gcs':True,
  'use_tpu':False
}

train_job_copy = {
  'name':'training_4',
  'source_training':'training_3',
  'bucket':'unittestimagerie0',
  'model':'faster_rcnn_resnet101_coco',
  'num_train_steps':100,
  'prep_name':'data3',
  'max_dim':1920,
  'min_dim':1200,
  'batch_size':1,
  'ml_engine':True,
  'ml_workers':16,
  'use_gcs':True,
  'use_tpu':False
}

gcs_storage = {
  'BUCKET':'unittestimagerie0',
  'USE_GCS':True
}
