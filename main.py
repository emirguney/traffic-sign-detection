import numpy as np
from src.data.prep_data import create_train_data, create_test_data
from src.model.test_model import test_model, save_accuracy
from src.model.model import build_model, train_model
from src.model.predict_model import predict

from tensorflow.keras.models import load_model
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    #create_train_data()
    #create_test_data()

    #x_train = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/x_train.npy')
    #y_train = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/y_train.npy')
    #x_val = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/x_val.npy')
    #y_val = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/y_val.npy')

    model_name = 'model_1'
    model_dir = os.path.join('C:/Users/desktop-21/PycharmProjects/deneme_proje/models', model_name)

    model = model_name
    modelname = test_model(model)



    image_dir = "C:/Users/desktop-21/Desktop/test_image.png"
    result = predict(modelname, image_dir)
    image_dir1 = "C:/Users/desktop-21/Desktop/cocuk.png"
    result1 = predict(modelname, image_dir1)
    image_dir2 = "C:/Users/desktop-21/Desktop/yaya.jpg"
    result2 = predict(modelname, image_dir2)

    print(result)
    print(result1)
    print(result2)