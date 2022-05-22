import json
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model


def predict(model_name, image_path):
    model_name ='model_1.h5'
    model_dir = os.path.join('C:/Users/desktop-21/PycharmProjects/deneme_proje/models', model_name)

    with open('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/classes.json') as json_file:
        classes = json.load(json_file)

    model = load_model(model_dir)

    image_array = cv2.imread(image_path)
    image_array = cv2.resize(image_array, (64, 64))
    image_array = np.expand_dims(image_array, axis=0)

    sign = model.predict_classes(image_array)[0]

    return classes[f'{sign + 1}']



