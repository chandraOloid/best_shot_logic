from flask import Flask, request, render_template, url_for, redirect, make_response, send_from_directory
from flask_restful import Resource, Api, reqparse
from detect_face_parts import *
import json

app = Flask(__name__)
api = Api(app)

class BestShotLogic(Resource):
    def get(self):
        return {
            'Oloid AddImg': {
                'usage': ['POST image with photo to add image to database',
                ]
            }
        }
    def post(self):
        if 'imgRec' in request.files:
            photo = request.files['imgRec']
            if photo.filename != '':
                if photo.filename.endswith('.png'):
                    img = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_COLOR)
                else: 
                    img = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                res = bestShotImg(img)
                return res

api.add_resource(BestShotLogic, '/')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

