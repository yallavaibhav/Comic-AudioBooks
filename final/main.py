import pickle
import re
from datetime import datetime
import os
from flask import Flask, render_template, request

import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
# lib for detection
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from engine import train_one_epoch, evaluate
# import utils
# import transforms as T
from PIL import Image
from torch_snippets import Report
import utils
from utils import panels
from utils2 import speech

app = Flask(__name__)

UPLOAD_FOLDER = "C:/Users/yalla/PycharmProjects/comic_test_deploy"


# model = pickle.load(open('C:/Users/saksh/data298/deploy/models/model.pkl','rb'))


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)

            pred = panels(image_location)
            pred2 = speech(pred)

            # pred = predict_func(image_location, MODEL) #image path and model input
            return render_template("index.html", Prediction=pred2)

    return render_template("index.html", Prediction="reuly")


# Panels -> panel files
# panel files -> character prediction code -> coordinates
# panel file -> speech bubbles prediction -> coordinates
# connect character with speech bubble
# update dataframe
#


if __name__ == "__main__":
    # app.run(port = 5000, debug = True)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
