import numpy
import numpy as np
import cv2
from random import randint
from rl_project.config import *


# noinspection DuplicatedCode
def create_img(x_cube_in_img, y_cube_in_img):
    cube = cubes[randint(0, len(cubes) - 1)]
    new_cube = np.zeros(background.shape)
    x_offset = int(x_cube_in_img - (cube.shape[0] / 2))
    y_offset = int(y_cube_in_img - (cube.shape[1] / 2))

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

    if debug:
        cv2.imwrite('images/overlay.png', final)
    return final


def create_binary_img(x_cube_in_img, y_cube_in_img):
    cube = cubes[randint(0, len(cubes) - 1)]
    new_cube = np.zeros(background.shape)
    x_offset = int(x_cube_in_img - (cube.shape[0] / 2))
    y_offset = int(y_cube_in_img - (cube.shape[1] / 2))

    new_cube[y_offset:y_offset + cube.shape[0], x_offset:x_offset + cube.shape[1]] = cube

    # create overlay image
    w, h, c = background.shape
    final = np.zeros((w, h, 1), )
    for iw in range(w):
        for ih in range(h):
            if not new_cube[iw][ih].any():
                final[iw][ih] = 0
            else:
                final[iw][ih] = 255

    if debug:
        cv2.imwrite('images/overlay.png', final)
    return final


def create_multiimage(locations):
    new_cube = np.zeros(background.shape)
    for loc in locations:
        cube = cubes[randint(0, len(cubes) - 1)]
        x_middle, y_middle = loc

        x_offset = int(x_middle - (cube.shape[0] / 2))
        y_offset = int(y_middle - (cube.shape[1] / 2))

        new_cube[y_offset:y_offset + cube.shape[0], x_offset:x_offset + cube.shape[1]] = cube

    # create overlay img
    # noinspection DuplicatedCode
    final = np.zeros(background.shape)
    w, h, c = background.shape
    for iw in range(w):
        for ih in range(h):
            if not new_cube[iw][ih].any():
                final[iw][ih] = background[iw][ih]
            else:
                final[iw][ih] = new_cube[iw][ih]
    if debug:
        cv2.imwrite('new_overlay.png', final)

    final = cv2.resize(final, (int(h / resize_factor), int(w / resize_factor)))
    return final