import cv2
import pickle
import numpy as np
from random import randint
from rl_project.config import *
from rl_project.image_creator import *

def create_dataset(size):
    c = 50
    labels = []

    for i in range(size):
        x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
        cv2.imwrite('dataset/' + str(i) + '.png', create_img(x, y))
        cv2.imwrite('dataset_binary/' + str(i) + '.png', create_binary_img(x, y))
        labels.append((y, x))
    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    create_dataset(5)
