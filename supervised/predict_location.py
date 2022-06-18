import numpy as np
import cv2
import keras
import imutils
from random import randint

from rl_project.config import resize_factor, background, base_model_path
from rl_project.supervised.create_dataset import create_binary_img
from rl_project.util.numpy_lru_cache_decorator import np_cache

x_model = keras.models.load_model(base_model_path + '/x_model.h5')
y_model = keras.models.load_model(base_model_path + '/y_model.h5')


# @np_cache
def predict_from_img(img):
    img = imutils.resize(img, width=int(img.shape[1] / resize_factor))
    img = np.array([img])

    x_pred = x_model(img)[0]
    y_pred = y_model(img)[0]
    return np.argmax(x_pred) * resize_factor, np.argmax(y_pred) * resize_factor


def predict(x, y):
    x_pred, y_pred = predict_from_img(create_binary_img(x, y))
    print(f'x: Real: {x}, Predicted: {x_pred}')
    print(f'y: Real: {y}, Predicted: {y_pred}', end='\n\n')


if __name__ == '__main__':
    c = 20
    for _ in range(5):
        x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
        predict(x, y)
