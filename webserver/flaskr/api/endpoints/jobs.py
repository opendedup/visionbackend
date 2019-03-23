import logging


from flask import request
from flask_restplus import Resource, Namespace, fields


import json


from flask import jsonify
from flask import make_response
from flask import g
from flask import request
from flask import Response



import settings


log = logging.getLogger(__name__)

ns = Namespace('jobs', description='List Job Information')

def list_jobs(fldr):
    job_ids = settings.storage.list_files(fldr,metadata=True)
    
    ar = []
    for jb in job_ids:
        nm = jb['name'][len(fldr):len(jb['name'])-5]
        nma = nm.split('_')
        status = 'done'
        if nma[3]=='f':
            status = 'failed'
        if nma[3]== 'r':
            status = 'running'
        job = {
            'name': nm,
            'start': int(nma[0]),
            'end':int(nma[1]),
            'type': nma[2],
            'status': status,
            'job_id':nma[4]}
        ar.append(job)
    return ar


@ns.route('/<string:id>')
class QJob(Resource):
    @ns.response(200, 'Returns a job status from queue')
    def get(self,id):
        """
        Returns job metadata for a given id.
        """
        jbstr = None
        jbstr = settings.storage.download_to_string('jobs/all/{}.json'.format(id))
        if jbstr is None:
            resp = {'no_such_job_id': id}
            return resp, 404
        else:
            jb = json.loads(jbstr)
            return jb,200


@ns.route('/running')
class RunningJobs(Resource):
    @ns.response(200, 'Returns running jobs from the queue')
    def get(self):
        """
        Returns a list of running jobs.
        """
        fldr = 'jobs/running/'
        return list_jobs(fldr),200

@ns.route('/failed')
class FailedJobs(Resource):
    @ns.response(200, 'Returns failed jobs from the queue')
    def get(self):
        """
        Returns a list of failed jobs.
        """
        fldr = 'jobs/failed/'
        
        return list_jobs(fldr),200

@ns.route('/finished')
class FinishedJobs(Resource):
    @ns.response(200, 'Returns finished jobs from the queue')
    def get(self):
        fldr = 'jobs/finished/'
        
        return list_jobs(fldr),200

@ns.route('/all')
class FinishedJobs(Resource):
    @ns.response(200, 'Returns finished jobs from the queue')
    def get(self):
        fldr = 'jobs/all/'
        
        return list_jobs(fldr),200
