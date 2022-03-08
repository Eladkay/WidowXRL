import numpy as np
import cv2
from random import randint


def create_img(background, cubes, x, y):
    cube = cubes[randint(0, len(cubes) - 1)]
    new_cube = np.zeros(background.shape)
    x_offset = int(x - (cube.shape[0] / 2))
    y_offset = int(y - (cube.shape[1] / 2))

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

    # remove in training
    cv2.imwrite('images/overlay.png', final)
    return final


if __name__ == '__main__':
    from config import background, cubes
    c = 50

    x, y = randint(c, background.shape[1] - c), randint(c, background.shape[0] - c)
    print(f'x: {x}, y: {y}')
    create_img(background, cubes, x, y)
