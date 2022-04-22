import cv2
import pickle
import numpy as np
from random import randint
from rl_project.config import *
from rl_project.image_creator import create_img, create_binary_img


def create_dataset(size, c, notify=0):
    labels = []

    for i in range(size):
        x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
        cv2.imwrite('dataset/' + str(i) + '.png', create_img(x, y))
        cv2.imwrite('dataset_binary/' + str(i) + '.png', create_binary_img(x, y))
        labels.append((y, x))
        if notify > 0 and i % notify == 0:
            print('{}/{}: {}, {}'.format(i, size, x, y))
    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    create_dataset(10000, 20, notify=50)
