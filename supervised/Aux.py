
RESIZE_FACTOR = 4
BASE = '../images/'
background = cv2.imread(BASE + 'background.png')
cubes = [cv2.imread(BASE + 'final_cube1.png'), cv2.imread(BASE + 'final_cube2.png'),
         cv2.imread(BASE + 'final_cube3.png')]


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
    return final


def create_binary_img(x_img, y_img):
    cube = cubes[randint(0, len(cubes) - 1)]
    new_cube = np.zeros(background.shape)
    x_offset = int(x_img - (cube.shape[0] / 2))
    y_offset = int(y_img - (cube.shape[1] / 2))

    new_cube[y_offset:y_offset + cube.shape[0], x_offset:x_offset + cube.shape[1]] = cube

    # create overlay img
    w, h, _ = background.shape
    final = np.zeros((w, h, 1), )
    w, h, c = background.shape
    for iw in range(w):
        for ih in range(h):
            if not new_cube[iw][ih].any():
                final[iw][ih] = 0
            else:
                final[iw][ih] = 255

    return final
