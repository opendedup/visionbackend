swagger: '2.0'
consumes:
  - application/json
info:
  title: Flexible Vision Preparation and Training API Gateway
  license:
    name: GPL V3
    url: https://www.gnu.org/licenses/gpl-3.0.en.html
  version: '1.0'
  description: Do Great things
  contact:
    name: Flexible Vision Github Site
    url: https://github.com/flexiblevision/visionbackend
    email: sam.silverberg@gmail.com
host: "${api_name}"
schemes:
# Uncomment the next line if you configure SSL for this API.
  - "https"
  - "http"  
x-google-endpoints:
    - name: "${api_name}"
      target: "${instance_fqdn}"
definitions:
  Export_Job:
    type: object
    properties:
      deploy_ml_engine:
        default: false
        type: boolean
        description: Deploy in Google ML Engine for prediction.
      training:
        type: string
        description: The training job to start with
        example: training0
      bucket:
        type: string
        description: The s3 bucket target
        example: bucket0
      model_name:
        type: string
        description: Use GCS for storage.
        example: model3
      project_name:
        type: string
        description: The name of the project that the model was trained against
        example: project0
      use_gcs:
        default: false
        type: boolean
        description: Use GCS for storage.
    required:
      - model_name
      - project_name
      - training
  Prep_Job:
    type: object
    properties:
      train_samples:
        default: 1000
        type: integer
        description: The number augemented training samples to create per image.
        example: 1000
      desired_size:
        default: 0
        type: integer
        description: 'The disired square dimensions with padding in pixels. If set to 0, this is ignored'
        example: 640
      aug_rules:
        description: Image Augmentation rules used for image augmentation
        allOf:
          - $ref: '#/definitions/Augmentation_Rules'
      name:
        type: string
        description: The name of the prep corpus that will be returned from the prep job
        example: data0
      test_percentage:
        default: 0
        type: number
        description: The percentage of the corpus to use for test data.
        example: 0.2
      aug:
        default: true
        type: boolean
        description: 'If true data will be augmented as part of prep. '
      test_samples:
        default: 200
        type: integer
        description: The number augemented testing samples to create per image. Testing samples should be 20% of training samples.
        example: 200
      project_name:
        type: string
        description: The name of the project to us as the corpus source.
        example: project0
      use_gcs:
        default: false
        type: boolean
        description: Use GCS for storage.
      bucket:
        type: string
        description: The s3 bucket target
        example: bucket0
    required:
      - name
      - project_name
  Augmentation_Rules:
    type: object
    properties:
      rotate:
        description: Rotate objects clockwise and counterclockwise
        items:
          type: integer
        type: array
        example:
          - 90
          - -90
        maxItems: 2
        minItems: 2
      scale:
        default: false
        type: boolean
        description: 'Randomly add slightly scale the image randomly between 0.8x - 1x. '
      crop:
        default: false
        type: boolean
        description: 'Randomly crop images. '
      hflip:
        default: 0
        type: number
        description: Horizontally Flip and Image a designated percentage of the time.
        example: 0.5
      shear:
        default: false
        type: boolean
        description: 'Randomly Slightly shear the image. '
      remove_out_of_image:
        default: cut_out_of_image
        type: string
        description: |-
          Specify "cut_out_partly" to cut out objects that are partly out of view.
              Specify "cut_out_of_image" to trim the box to the image field of view. Specify "leave_partly_in_image" to keep all boxes regardless of their field of view.
               
        example: cut_out_of_image
        enum:
          - leave_partly_in_image
          - cut_out_partly
          - cut_out_of_image
      noise:
        default: false
        type: boolean
        description: 'Randomly add noise to images. '
      vflip:
        default: 0
        type: number
        description: Vertically Flip and Image a designated percentage of the time.
        example: 0.5
      contrast:
        default: false
        type: boolean
        description: 'Randomly add contrast to image. '
  Train_Job:
    type: object
    properties:
      prep_name:
        type: string
        description: The corpus to train against
        example: data0
      max_dim:
        type: integer
        description: The maximum length of the largest dimension of the image. If the image's width/heigh is larger than this dimension is will be resized to max_dim
        example: 1280
      mle_region:
        default: us-central1
        type: string
        description: The region to perform ml engine trainings in.
        example: us-central1
      parameter_servers:
        default: 0
        type: integer
        description: The number of parameter servers to bring up with the ml engine model. If 0 specified it will automatically create the correct number.
      name:
        type: string
        description: The name of the model that will be trained
        example: project0
      ml_engine:
        default: true
        type: boolean
        description: If true use mlengine for training
      batch_size:
        default: 2
        type: integer
        description: The number of images to process in training in one pass. A larger number is increases speed at the expense of memory.
        example: 2
      use_gcs:
        default: false
        type: boolean
        description: Use GCS for storage.
      source_training:
        default: ''
        type: string
        description: A previous training to use as the basis for this training.
        example: training0
      use_tpu:
        default: false
        type: boolean
        description: Use a Google Tensor Processing Unit (TPU).
      bucket:
        type: string
        description: The s3 bucket target
        example: bucket0
      model:
        type: string
        description: The pretrained model to start with
        example: faster_rcnn_inception_v2_coco
        enum:
          - faster_rcnn_inception_v2_coco
          - ssd_resnet50_v1_fpn_shared_box_predictor_coco14_sync
          - ssd_mobilenet_v1_fpn_shared_box_predictor_coco14_sync
          - faster_rcnn_resnet101_coco
          - faster_rcnn_inception_resnet_v2_atrous_coco
          - faster_rcnn_resnet50_coco
      min_dim:
        type: integer
        description: The maximum length of the smallest dimension of the image. If the image's width/heigh is larger than this dimension is will be resized to min_dim
        example: 720
      ml_workers:
        default: 8
        type: integer
        description: The number of workers to use for an ml engine training
        example: 8
      num_train_steps:
        type: integer
        description: The number of steps to train
        example: 50000
    required:
      - batch_size
      - max_dim
      - min_dim
      - model
      - name
      - num_train_steps
      - parameter_servers
      - prep_name
responses:
  MaskError:
    description: When any error occurs on mask
  ParseError:
    description: When a mask can't be parsed
produces:
  - application/json
basePath: /api/process
tags:
  - description: Prepare a vision corpus for training
    name: prep
  - description: Train a vision corpus on a prepared dataset
    name: train
paths:
  '/prep/job/{id}':
    get:
      responses:
        '200':
          description: Returns a job status from queue
      summary: Returns job metadata for a given id
      tags:
        - prep
      operationId: get_prep_job
      security:
        - api_key: []
    
    parameters:
      - type: string
        name: id
        in: path
        required: true
  /prep/jobs/failed:
    get:
      responses:
        '200':
          description: Returns failed jobs from the queue
      summary: Returns a list of failed jobs
      tags:
        - prep
      security:
        - api_key: []
      operationId: get_prep_failed_jobs
  /prep/jobs/finished:
    get:
      responses:
        '200':
          description: Returns finished jobs from the queue
      tags:
        - prep
      security:
        - api_key: []
      operationId: get_prep_finished_jobs
  /prep/jobs/running:
    get:
      responses:
        '200':
          description: Returns running jobs from the queue
      summary: Returns a list of running jobs
      security:
        - api_key: []
      tags:
        - prep
      operationId: get_prep_running_jobs
  /prep/run:
    post:
      responses:
        '201':
          description: '{"status":"queued","job_id":"uuid"}'
      summary: Executes a prep job to create an image corpus for training
      description: Use this method to start a prep job.
      security:
        - api_key: []
      parameters:
        - name: payload
          required: true
          in: body
          schema:
            $ref: '#/definitions/Prep_Job'
      operationId: post_run_prep
      tags:
        - prep
  /train/export:
    post:
      responses:
        '201':
          description: '{"status":"queued","job_id":"uuid"}'
      summary: Exports a finished training for prediction
      description: Use this method to export a finished training job for prediction.
      parameters:
        - name: payload
          required: true
          in: body
          schema:
            $ref: '#/definitions/Export_Job'
      operationId: post_export_training
      tags:
        - train
      security:
        - api_key: []
  '/train/job/{id}':
    get:
      responses:
        '200':
          description: Returns a job status from queue
      tags:
        - train
      operationId: get_train_q_job
      security:
        - api_key: []
    parameters:
      - type: string
        name: id
        in: path
        required: true
    
  /train/jobs/failed:
    get:
      responses:
        '200':
          description: Returns failed jobs from the queue
      summary: Returns a list of failed jobs
      tags:
        - train
      operationId: get_train_failed_jobs
      security:
        - google_id_token: []
  /train/jobs/finished:
    get:
      responses:
        '200':
          description: Returns finished jobs from the queue
      tags:
        - train
      operationId: get_train_finished_jobs
      security:
        - api_key: []
  /train/jobs/running:
    get:
      responses:
        '200':
          description: Returns runing jobs from the queue
      summary: Returns a list of running jobs
      tags:
        - train
      operationId: get_train_running_jobs
      security:
        - api_key: []
  /train/run:
    post:
      responses:
        '201':
          description: '{"status":"queued","job_id":"uuid"}'
      summary: Executes a training
      description: Use this method to start a training.
      parameters:
        - name: payload
          required: true
          in: body
          schema:
            $ref: '#/definitions/Train_Job'
      operationId: post_run_training
      tags:
        - train
      security:
        - api_key: []
securityDefinitions:
  # This section configures basic authentication with an API key.
  api_key:
    type: "apiKey"
    name: "key"
    in: "query"
  # This section configures authentication using Google API Service Accounts
  # to sign a json web token. This is mostly used for server-to-server
  # communication.
  google_id_token:
    type: oauth2
    authorizationUrl: ""
    flow: implicit
    x-google-issuer: "https://accounts.google.com"
    x-google-jwks_uri: "https://www.googleapis.com/oauth2/v1/certs"
