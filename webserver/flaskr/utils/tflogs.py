import tensorflow as tf
import json
import os
from pathlib import Path

import cv2
from PIL import Image
import io
import base64

def get_events(cloud_folder,temp_dir,storage_controller,full=False,dim=720):
    
    flslt = storage_controller.list_files(cloud_folder,metadata=True)
    ml = []
    for f in flslt:
        if 'cmle-training-master' in f['name']:
            ml.append(f)
    ml.sort(key=sortmle,reverse=True)
    events = []
    ek = []
    if len(ml) > 0:
        dfd = os.path.dirname(os.path.abspath(temp_dir + '/' + ml[0]['name']))
        dfl = temp_dir + '/' + ml[0]['name']
        os.makedirs(dfd,exist_ok=True)
        logfile = Path(dfl)
        if not logfile.is_file():
            storage_controller.download_to_file(ml[0]['name'],dfl)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for e in tf.train.summary_iterator(dfl):
                if hasattr(e,'summary'):
                    ek.append(e)
                    evt = {}
                    evt['timestamp'] = int(e.wall_time*1000)
                    evt['step'] = e.step
                    evt['summary'] = []
                    for v in e.summary.value:
                        if hasattr(v,'simple_value'):
                            value = {}
                            value['key'] = v.tag
                            value['value'] = v.simple_value
                            evt['summary'].append(value)
                        if full and hasattr(v,'image') and len(v.image.encoded_image_string) > 0:
                            imt = v.image
                            img = tf.image.decode_png(imt.encoded_image_string,imt.colorspace).eval()
                            img = image_resize(img,width=dim)
                            ret,_img = cv2.imencode('.jpg',img)
                            image = Image.open(io.BytesIO(_img.tobytes()))
                            imgByteArr = io.BytesIO()
                            image.save(imgByteArr, format='JPEG')
                            imgByteArr = imgByteArr.getvalue()
                            ncoded_string = base64.b64encode(imgByteArr).decode("utf-8")
                            value = {}
                            value['key'] = v.tag
                            value['image_b64'] = ncoded_string
                            evt['summary'].append(value)

                    if len(evt['summary']) > 0:
                        events.append(evt)
        if len(events) == 0:
            os.remove(dfl)
    return events

def sortmle(e):
    return e['lastModified']

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


