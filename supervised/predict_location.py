import numpy as np
import cv2
import keras
import imutils
from random import randint
from rl_project.config import resize_factor, background, base_model_path
from rl_project.config import cf as c
from rl_project.supervised.create_dataset import create_binary_img
from rl_project.util.numpy_lru_cache_decorator import np_cache
x_model = keras.models.load_model(base_model_path + '/x_model.h5')
y_model = keras.models.load_model(base_model_path + '/y_model.h5')


def yagel_predict(param):
    """
        I was having problems with the supervised model, so I asked around and explained the problem.
        A guy called Yagel said "Why don't you just go over the image and check where the white pixels are?"
        "Not everything has to involve machine learning", he said. I scoffed. But then I thought, "Why not?"
    """
    white_pixels = []
    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            if param[i, j].max() > 0:
                white_pixels.append((i, j))
    top_x, top_y, bottom_x, bottom_y = 0, 0, 0, 0
    for i in range(len(white_pixels)):
        if white_pixels[i][0] < top_x:
            top_x = white_pixels[i][0]
        if white_pixels[i][1] < top_y:
            top_y = white_pixels[i][1]
        if white_pixels[i][0] > bottom_x:
            bottom_x = white_pixels[i][0]
        if white_pixels[i][1] > bottom_y:
            bottom_y = white_pixels[i][1]
    return (top_y + bottom_y), (top_x + bottom_x)


# @np_cache
def predict_from_img(img):
    img = imutils.resize(img, width=int(img.shape[1] / resize_factor))
    img = np.array([img])

    x_pred = x_model(img)[0]
    y_pred = y_model(img)[0]
    return np.argmax(x_pred) * resize_factor, np.argmax(y_pred) * resize_factor


def predict(x, y):
    img = create_binary_img(x, y)
    x_pred, y_pred = predict_from_img(img)
    x_pred_yagel, y_pred_yagel = yagel_predict(img)
    print(f'x: Real: {x}, Predicted by supervised: {x_pred}, Predicted by Yagel: {x_pred_yagel}')
    print(f'y: Real: {y}, Predicted by supervised: {y_pred}, Predicted by Yagel: {y_pred_yagel}', end='\n\n')


def tweak_params(count):
    s_x = ""
    s_y = ""
    for _ in range(count):
        x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
        x_pred_yagel, y_pred_yagel = yagel_predict(create_binary_img(x, y))
        s_x += f"{x},{x_pred_yagel}\n"
        s_y += f"{y},{y_pred_yagel}\n"
    fx = open("x_params.csv", "w")
    fy = open("y_params.csv", "w")
    fx.write(s_x)
    fy.write(s_y)
    fx.close()
    fy.close()


if __name__ == '__main__':
    tweak_params(100)
    # for _ in range(15):
    #     x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
    #     predict(x, y)
