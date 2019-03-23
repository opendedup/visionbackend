jobs =[ {
  "use_gcs": False,
  "aug_rules": {
    "remove_out_of_image": "cut_out_of_image",
    "rotate": [180,-180]
  },
  "desired_size":1024,
  "aug": True,
  "train_samples": 100,
  "test_samples": 20,
  "bucket": "bucket0",
  "project_name": "gummies2",
  "process_json":True,
  "access_key":"imagerie",
  "secret_key":"imagerie",
  "url":"http://localhost:9002",
  "name": "data0"
},{
  "use_gcs": False,
  "desired_size":1024,
  "aug": False,
  "train_samples": 10,
  "test_samples": 2,
  "bucket": "bucket0",
  "project_name": "gummies2",
  "process_json":True,
  "access_key":"imagerie",
  "secret_key":"imagerie",
  "url":"http://localhost:9002",
  "name": "data1"
}]

s3_storage = {
  'BUCKET':'bucket0',
  'S3_URL':'http://localhost:9002',
  'ACCESS_KEY':'imagerie',
  'SECRET_KEY':'imagerie',
  'USE_GCS':False
}
gcs_storage = {
  'BUCKET':'unittestimagerie0',
  'USE_GCS':True
}
