import os
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "mobilenet_fresh_rotten_softmax.h5"
MODEL_URL = "PASTE_GOOGLE_DRIVE_DIRECT_LINK"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
classes = ["Fresh", "Rotten"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    return classes[idx], round(float(preds[idx]) * 100, 2)
