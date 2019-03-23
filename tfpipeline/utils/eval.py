import tensorflow as tf
import json

output_filepath = "/home/sam/Downloads/training_jobs_gummies2_04ff468b-226b-4c99-bb87-f478d56369d3_eval_0_events.out.tfevents.1552922687.cmle-training-master-32e09be4e0-0-w4jgx"

events = []

for e in tf.train.summary_iterator(output_filepath):
    if hasattr(e,'summary'):
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
        if len(evt['summary']) > 0:
            events.append(evt)
print(json.dumps(events))

