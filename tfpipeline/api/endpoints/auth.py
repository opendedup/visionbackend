import logging

from flask import request
from flask_restplus import Resource, Namespace, fields

import json

from pathlib import Path

from datetime import datetime as newdt
import os

from flask import make_response
from flask import g
from flask import request
from flask import Response

from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)

ns = Namespace('auth', description='Authenication Components')

login_creds = ns.model('Login_Credientials', {
    'username': fields.String(required=True,
                          description='The username to login with',
                          example='admin'),
    'password':fields.String(required=True, description='The password',
                           example='admin')
})

token_resp = ns.model('Auth_Token_Response',{
    'access_token':fields.String(required=True,
                          description='The JWT Token',
                          example='Azbk')
})
user = None
pwd = None
if 'USER_NAME' in os.environ:
       user = os.environ['USER_NAME']
else:
    user = 'admin'
if 'PASSWORD' in os.environ:
    pwd = os.environ['PASSWORD']
else:
    pwd = 'admin'


@ns.route('/login')
@ns.doc(security=None)
class LoginResponse(Resource):
    @ns.response(201, '{"status":"queued","job_id":"uuid"}')
    @ns.response(401, '{"msg": "Bad username or password"}')
    @ns.expect(login_creds)
    @ns.marshal_with(token_resp)
    def post(self):
        username = request.json.get('username', None)
        password = request.json.get('password', None)
        if username != user or password != pwd:
            return {"msg": "Bad username or password"}, 401

        access_token = create_access_token(identity=username)
        print(access_token)
        return {'access_token':access_token}, 201