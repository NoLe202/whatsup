from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from tensorflow.python.keras.models import load_model
from keras.preprocessing import image
from keras import applications

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from gevent.pywsgi import WSGIServer

import tensorflow as tf


config = config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)


app = Flask(__name__)

MODEL_PATH = 'mymodel-2.h5'

model = load_model(MODEL_PATH)

def inception_architecture():
    model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(80, 80, 3))
    return model

model_architecture = inception_architecture()

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def model_predict(img_path):
    image_to_predict = path_to_tensor(img_path).astype('float32')
    model = model_architecture
    model = load_model('mymodel-2.h5')
    pred = model.predict(image_to_predict)
    if np.argmax(pred) == 0:
        return [1., 0., pred[0][0]]
    elif np.argmax(pred) == 1:
        return [0., 1., pred[0][0]]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        preds = model_predict(file_path)
        result = preds[2] *100
        print(round(result,2))
        return str(round(result,2))
    return None


if __name__ == '__main__':
    app.run(debug=True)