import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def visualize_model_scores():
    df = pd.read_csv("C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/scores.csv")

    ax = df.plot.barh(
        x="model",
        y="score",
        figsize=(10, 9),
        fontsize=12,
        title="Model Accuracies",
    )
    ax.set_xlabel("Scores")
    ax.set_ylabel("Models")
    plt.savefig("C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/figures/model_scores.png")
    plt.close()


def visualize_history(model_name):
    history = np.load('C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/models_histories/' + model_name +
                      '_history.npy', allow_pickle='TRUE').item()

    if "history" in history:
        history = history.history

    plt.style.use('default')
    plt.style.use('seaborn')

    plt.figure(0)
    plt.plot(history['accuracy'], label='train accuracy')
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/figures/' + model_name + '_train_accuracy')

    plt.figure(1)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('C:/Users/desktop-21/PycharmProjects/deneme_proje/reports/figures/' + model_name + '_train_loss')


