# -*- coding: utf-8 -*-
from flask import Flask
from flask_restful import Resource, Api, reqparse, fields, marshal_with
import time
from resources.meshrecommend import MeshRecommend

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        time.sleep(10)
        return {'hello': 'world'}

class Test(Resource):
    def get(self):
        #time.sleep(10)
        return '123456test123456'

class Test2(Resource):
    def get(self, arg):
    #time.sleep(10)
        print arg
        return '123456test123456'

api.add_resource(HelloWorld, '/')
api.add_resource(Test, '/test/')
api.add_resource(Test2, '/test2/<string:arg>')
api.add_resource(MeshRecommend, '/mesh/<string:arg>')

if __name__ == '__main__':
    app.run(host="172.16.1.20", port=8023)
