import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

import os
import shutil
import random

BASE_DIR = "fruit"
TEST_DIR = os.path.join(BASE_DIR, "test")

classes = ["fresh", "rotten"]
NUM_IMAGES = 20  

for cls in classes:
    src_dir = os.path.join(BASE_DIR, cls)
    dst_dir = os.path.join(TEST_DIR, cls)

    os.makedirs(dst_dir, exist_ok=True)

    images = os.listdir(src_dir)
    random.shuffle(images)

    for img in images[:NUM_IMAGES]:
        shutil.move(
            os.path.join(src_dir, img),
            os.path.join(dst_dir, img)
        )

print("Images moved to test folder successfully")

model = load_model("mobilenet_fresh_rotten_softmax.h5")

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "fruit/test",
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes

print(classification_report(
    y_true,
    y_pred,
    target_names=test_gen.class_indices.keys()
))


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("mobilenet_fresh_rotten_softmax.h5")

img_path = "C:/Users/kandi/OneDrive/Desktop/FruitProject/real_image4.webp"

IMG_SIZE = 224
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)


pred = model.predict(x)[0]  

class_labels = ["fresh", "rotten"]

for i, label in enumerate(class_labels):
    print(f"{label}: {pred[i]*100:.2f}%")




