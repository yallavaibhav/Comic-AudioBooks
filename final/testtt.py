import pickle
import re
from datetime import datetime
import os
from flask import Flask, render_template, request, session

import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
#lib for detection



from PIL import Image

import utils
from utils import panels
from utils2 import speech


app = Flask(__name__)

UPLOAD_FOLDER = "C:/Users/yalla/PycharmProjects/Comic_final1/static"

app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)

            # pred = panels(image_location)
            pred = panels(image_location)  # give file path
            session['pred_val'] = speech(pred)
            # img_file_path = session.get('pred', None)
            return render_template('index.html')
            # return render_template("index.html", Prediction = pred)

            # #pred = predict_func(image_location, MODEL) #image path and model input
            # return render_template("index.html", Prediction = pred)
    return render_template("index.html", Prediction="result")


@app.route('/show_image')
def show_image():
    # img_file_path = session.get('pred_val', None)
    # Display image in Flask application web page
    # return render_template('display.html', user_image=img_file_path)
    return render_template('display.html')


if __name__ == "__main__":
    app.run(port=5000, debug=True)