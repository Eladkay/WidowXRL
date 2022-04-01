import numpy as np
import cv2
import keras
import imutils
from random import randint
from Aux import *

x_model = keras.models.load_model('x_model.h5')
y_model = keras.models.load_model('y_model.h5')


def predict(x, y):
    cv2.imwrite('predict.png', create_binary_img(x, y))
    img = cv2.imread('predict.png') / 255
    img = imutils.resize(img, width=int(img.shape[1] / RESIZE_FACTOR))
    img = np.array([img])

    x_pred = x_model(img)[0]
    y_pred = y_model(img)[0]

    print(f'x: Real: {x}, Predicted: {np.argmax(x_pred) * RESIZE_FACTOR}')
    print(f'y: Real: {y}, Predicted: {np.argmax(y_pred) * RESIZE_FACTOR})',
          end='\n\n')


if __name__ == '__main__':
    c = 50
    for _ in range(5):
        x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
        predict(x, y)
