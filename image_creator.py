import numpy as np
import cv2
from random import randint
from config import *


# noinspection DuplicatedCode
def create_img(x_img, y_img):
    cube = cubes[randint(0, len(cubes) - 1)]
    new_cube = np.zeros(background.shape)
    x_offset = int(x_img - (cube.shape[0] / 2))
    y_offset = int(y_img - (cube.shape[1] / 2))

    new_cube[y_offset:y_offset + cube.shape[0], x_offset:x_offset + cube.shape[1]] = cube

    # create overlay img
    final = np.zeros(background.shape)
    w, h, c = background.shape
    for iw in range(w):
        for ih in range(h):
            if not new_cube[iw][ih].any():
                final[iw][ih] = background[iw][ih]
            else:
                final[iw][ih] = new_cube[iw][ih]

    if write_image_to_file:
        cv2.imwrite('images/overlay.png', final)
    return final


def create_binary_img(x_img, y_img):
    cube = cubes[randint(0, len(cubes) - 1)]
    new_cube = np.zeros(background.shape)
    x_offset = int(x_img - (cube.shape[0] / 2))
    y_offset = int(y_img - (cube.shape[1] / 2))

    new_cube[y_offset:y_offset + cube.shape[0], x_offset:x_offset + cube.shape[1]] = cube

    # create overlay image
    w, h, c = background.shape
    final = np.zeros((w, h, 3), )
    for iw in range(w):
        for ih in range(h):
            for iz in range(3):
                if not new_cube[iw][ih].any():
                    final[iw][ih][iz] = 0
                else:
                    final[iw][ih][iz] = 255

    if write_image_to_file:
        cv2.imwrite('images/overlay.png', final)
    return final
