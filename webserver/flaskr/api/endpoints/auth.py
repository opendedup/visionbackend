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
import settings
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity,jwt_refresh_token_required,
    create_access_token,create_refresh_token,
    get_raw_jwt
)

ns = Namespace('auth', description='Authenication Components')
jwt = settings.jwt
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
                          example='Azbk'),
    'refresh_token':fields.String(required=False,
                          description='The JWT Refresh Token',
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
    @ns.response(200, '{"access_token":"token"}')
    @ns.response(401, '{"msg": "Bad username or password"}')
    @ns.expect(login_creds)
    @ns.marshal_with(token_resp)
    def post(self):
        username = request.json.get('username', None)
        password = request.json.get('password', None)
        if username != user or password != pwd:
            return {"msg": "Bad username or password"}, 401

        access_token = create_access_token(identity=username)
        refresh_token = create_refresh_token(identity=username)
        return {'access_token':access_token,'refresh_token': refresh_token}, 200

@ns.route('/verify')
@ns.doc(security='Bearer Auth')
class VerifyResponse(Resource):
    @ns.response(200, '{"logged_in_as":"current_user"}')
    @jwt_required
    def get(self):
        current_user = get_jwt_identity()
        return {'logged_in_as':current_user}, 200

@ns.route('/refresh')
@ns.doc(security=None)
class TokenRefresh(Resource):
    @jwt_refresh_token_required
    def post(self):
        current_user = get_jwt_identity()
        access_token = create_access_token(identity = current_user)
        return {'access_token': access_token}

@ns.route('/logout')
@ns.doc(security='Bearer Auth')
class Logout(Resource):
    @jwt_required
    def delete(self):
        """
        revokes the user's access token.
        """
        jti = get_raw_jwt()['jti']
        settings.blacklist.add(jti)
        return {"msg": "Successfully logged out"}, 200

@ns.route('/logout2')
@ns.doc(security='Bearer Auth')
class Logout2(Resource):
    @jwt_required
    def delete(self):
        """
        revokes the user's refresh token.
        """
        jti = get_raw_jwt()['jti']
        settings.blacklist.add(jti)
        return {"msg": "Successfully logged out"}, 200