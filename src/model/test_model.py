import os
import numpy as np
import csv
from os.path import exists
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save_accuracy(model_name, score):

    fields = ["model", "score"]
    row = [model_name, score]
    try:
        if exists("C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/scores.csv") == False:
            with open(r"C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/scores.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerow(row)
        else:
            with open(r"C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/scores.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    except:
        pass


def test_model(model_name):
    model_name = model_name+'.h5'
    model_dir = os.path.join('C:/Users/desktop-21/PycharmProjects/deneme_proje/models', model_name)
    x_test = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/x_test.npy')
    y_test = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/y_test.npy')
    model = load_model(model_dir)

    return model.evaluate(x_test, y_test)[1]



