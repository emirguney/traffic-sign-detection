import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def create_train_data(num_classes=43,
                      resize_col=64, resize_row=64,
                      test_size=0.2, random_state=1):

    train_data_dir = 'C:/Users/desktop-21/PycharmProjects/deneme_proje/data/raw/train'
    # Resim matrislerinin tutulacağı dizi.
    data = []
    # Resim sınıflarının tutulacağı dizi.
    labels = []
    # Klasör içerisinde bulunan her bir sınıf klasörünün dosya yolu.
    train_files = os.listdir(train_data_dir)

    # Her bir dosya yolunu dolaşacak döngü
    for classes in train_files:

        # Dosya yolunun adı aynı zamanda sınıf numaramız.
        classname = str(classes)
        path = os.path.join(train_data_dir, classname)
        # Dosyadan tüm resimlerin yolunu alıyoruz.
        images = os.listdir(path)

        # Tüm resimleri dolaşacak döngü
        for image in images:
            image_path = os.path.join(path, image)
            # Resim okunarak bir diziye aktarılıyor
            image_array = cv2.imread(image_path)
            # Resim yeniden boyutlandırılıyor.
            image_array = cv2.resize(image_array, (resize_row, resize_col))
            # Verilerin ilgili dizilere eklenmesi
            data.append(image_array)
            labels.append(classname)

        # Dizilerin numpy dizilerine dönüştürülmesi
    data = np.array(data)
    labels = np.array(labels)

    # Resim verilerinin train ve validation verisi olarak
    # bölünmesi işlemi
    x_train, x_val, y_train, y_val = train_test_split(
        data, labels, test_size=test_size, random_state=random_state)
    # Sınıfların kategorik matris hale getirilmesi
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    np.save('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/x_train.npy', x_train)
    np.save('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/y_train.npy', y_train)
    np.save('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/x_val.npy', x_val)
    np.save('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/processed/y_val.npy', y_val)

    #print(data.shape, labels.shape)

    #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)

def create_test_data(
        num_classes=43,
        resize_col=64, resize_row=64):
    dataset_dir = 'C:/Users/desktop-21/PycharmProjects/deneme_proje/data/raw'
    test_info_dir = 'C:/Users/desktop-21/PycharmProjects/deneme_proje/data/raw/test.csv'
    # Csv dosyasının okunması
    y_test = pd.read_csv(test_info_dir)
    # Resimlerin dosya yollarının csvden okunması
    images = y_test['Path'].values
    # Resimlerin sınıf bilgilerinin csvden okunması
    y_test = y_test["ClassId"].values

    # Resim matrislerinin tutulacağı dizi
    x_test = []

    # Test klasöründe bulunan tüm resimleri dolaşacak döngü
    for image in images:
        image_path = os.path.join(dataset_dir, image)
        # Resim okunarak bir diziye aktarılıyor
        image_array = cv2.imread(image_path)
        # Resim yeniden boyutlandırılıyor.
        image_array = cv2.resize(image_array, (resize_row, resize_col))
        # Verilerin ilgili diziye eklenmesi
        x_test.append(image_array)

        # Dizilerin numpy dizilerine dönüştürülmesi
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Sınıfların kategorik matris hale getirilmesi
    y_test = to_categorical(y_test, num_classes)
    np.save('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/x_test.npy', x_test)
    np.save('C:/Users/desktop-21/PycharmProjects/deneme_proje/data/y_test.npy', y_test)





