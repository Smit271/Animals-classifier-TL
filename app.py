from flask import Flask, render_template, url_for, flash
from flask.globals import request
from tensorflow.python.ops.gen_math_ops import Imag
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = 'sd51vs9v8v56sds@Wx32d2'
app.config["MODEL_PATH"] = "./model/model.h5"
app.config["UPLOAD_PATH"] = "./uploads/"

model = tf.keras.models.load_model(app.config["MODEL_PATH"])
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def get_output(img):
    img1 = cv2.imread(img)
    img2 = Image.fromarray(img1, 'RGB')
    resized_image = img2.resize((224, 224))
    scaled_image = np.array(resized_image) / 255.0
    test_img = np.expand_dims(scaled_image, axis=0)
    scores = model.predict(test_img)
    preds = np.argmax(scores, axis = 1)
    return f"{classes[preds[0]]}"


@app.route("/", methods = ["POST", 'GET'])
def home():
    if request.method == "POST":
        upload_image()
    render_template("index.html")


def upload_image():
    if request.files:
        output = "NO"
        in_image = request.files["in_image"]
        filename = str(time.time())
        in_image.save(os.path.join(app.config["UPLOAD_PATH"], secure_filename(f"{filename}.{in_image.filename.split('.')[-1]}")))
        if os.path.exists(os.path.join(app.config["UPLOAD_PATH"],secure_filename(f"{filename}.{in_image.filename.split('.')[-1]}"))):
            output = get_output(os.path.join(app.config["UPLOAD_PATH"], secure_filename(f"{filename}.{in_image.filename.split('.')[-1]}")))
            if output=="NO":
                flash(f'Please upload image first!', 'danger')
                return render_template("index.html")
            # print(output)
        if output!="NO":
            flash(f'This is photo of {output.capitalize()}!', 'success')
            return render_template("index.html")
    else:
        flash(f'Please upload image first!', 'danger')
        return render_template("index.html")
    return render_template('index.html')


if __name__ =="__main__":
    app.run(debug=True)