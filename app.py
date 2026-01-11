from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

MODEL_PATH = "mobilenet_fresh_rotten_softmax.h5"   
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224
CLASS_NAMES = ["Fresh", "Rotten"]   

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", label="No Image", confidence=0)

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", label="No Image", confidence=0)

    # save image temporarily
    img_path = "temp.jpg"
    file.save(img_path)

    # preprocess
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # predict
    predictions = model.predict(img_array)[0]

    confidence = float(np.max(predictions) * 100)
    label = CLASS_NAMES[np.argmax(predictions)]

    os.remove(img_path)

    return render_template(
        "index.html",
        label=label,
        confidence=round(confidence, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)

