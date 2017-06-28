###################################IMPORTING ALL THE LIBRARIES#################################################
import argparse
import base64
import json
import cv2 
import tensorflow as tf
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import load_model
##################################################################################################################

tf.control_flow_ops = tf
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
size1 = 64
size2 =96
prop1 = 40/160
propx1 = 10/320
propx2 = 310/320
model = load_model('model_9.h5')
@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_a = np.asarray(image)
    #Full color
    image_array =image_a[:,:,:].squeeze()
    crp1_y = int(prop1*image_array.shape[0])
    crp2_y = int(image_array.shape[0])
    crp1_x = int(propx1*image_array.shape[1])
    crp2_x = int(propx2*image_array.shape[1])
    crop_img = image_array[crp1_y:crp2_y,crp1_x:crp2_x,:] 
    image_array2= np.zeros((size1,size2,3))
    image_array2[:,:,:] = cv2.resize(crop_img, (size2,size1))
    transformed_image_array = image_array2[None, :, :, :]
	

	
    
    with tf.device("/cpu:0"):
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    if abs(steering_angle) < .1:
         throttle = 0.25
    else:
        throttle = 0.2       

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
