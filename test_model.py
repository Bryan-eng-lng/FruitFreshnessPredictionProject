from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("mobilenet_fresh_rotten_softmax.h5")

img_path = "C:/Users/kandi/OneDrive/Desktop/FruitProject/real_image2.jpg"

IMG_SIZE = 224
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

# Predict probabilities
pred = model.predict(x)[0]  


class_labels = ["fresh", "rotten"]

for i, label in enumerate(class_labels):
    print(f"{label}: {pred[i]*100:.2f}%")