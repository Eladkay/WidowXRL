from random import randint
import cv2
import numpy as np
from time import time


def distance(a, b):
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)) ** 0.5


EPSILON = 10
Divide_Factor = 1  # 8
Caution_Factor = 17


class Img_generator_grab_only:

    def __init__(self, green_area=None, num=3):
        self.original_num = num
        self.background = cv2.imread('background.jpeg')
        self.cubes = [cv2.imread('final_cube1.jpeg'), cv2.imread('final_cube2.png'), cv2.imread('final_cube3.png')]

        # the area where we want the cubes to be
        # format: [x_min, x_max, y_min, y_max]
        if green_area is None:
            self.green_area = [0, 320, 0, 215]
        else:
            self.green_area = green_area

        # the bounds of the board
        self.min_ = (40, 40)
        self.max_ = (280, 390)

        # generate the cubes locations and image
        self.locations = self.generate_random(num)
        self.image = self.generate_img()

    # returns img, if the grab was successful/drop is in green area and if all cubes are in the green area
    def go_to_location(self, location) -> (np.ndarray, (int, bool), bool):
        # ! if trying to grab a cube that is already in the green area IGNORE
        if self.is_in_green(location):
            return self.image, (self.GRAB, False), False
        
        for loc in self.locations:
            if distance(loc, location) < EPSILON:
                # remove cube
                self.locations.remove(loc)
                self.image = self.generate_img()
                return self.image, (self.GRAB, True), not len(self.locations)

        cv2.imwrite("current.png", self.image)

        # if grab was unsuccessful
        return self.image, (self.GRAB, False), False

    # num: number of cubes to be generated *optional*
    def reset(self, num=0):
        if num:
            self.original_num = num

        self.locations = self.generate_random(self.original_num)
        self.image = self.generate_img()
        return self.image

    def get_img(self):
        return self.image

    def action_bounds(self):
        return (self.min_[0], self.max_[0]), (self.min_[1], self.max_[1])

    def observation_bounds(self):
        return (0, self.background.shape[0]), (0, self.background.shape[1])

    def generate_random(self, num):
        min_, max_ = self.min_, self.max_
        min_distance = 50
        locs = []
        # check if area is only the right side of the board
        is_area_right_size = self.green_area[0] == 0 and self.green_area[2] == 0

        while num > 0:
            # generate random location for right side area
            if is_area_right_size:
                new_loc = (randint(self.green_area[3] + Caution_Factor, max_[1] - Caution_Factor), randint(min_[0] + Caution_Factor, max_[0] - Caution_Factor))
            # generate random location for any other area (slower)
            else:
                new_loc = (randint(min_[1], max_[1]), randint(min_[0], max_[0]))

                # we don't want the starting position to be in the green area
                if self.is_in_green(new_loc):
                    continue

            # check if the new location is too close to the previous ones
            fine = True
            for l in locs:
                if distance(new_loc, l) < min_distance:
                    fine = False

            if fine:
                locs.append(new_loc)
                num -= 1

        return locs

    def is_in_green(self, location):
        x, y = location
        return self.green_area[0] < y < self.green_area[1] and self.green_area[2] < x < self.green_area[3]

    def generate_img(self):
        background = self.background
        cubes = self.cubes
        locations = self.locations
        final = background

        new_cube = np.zeros(background.shape)
        for loc in locations:
            cube = cubes[randint(0, len(cubes) - 1)]
            x_middle, y_middle = loc

            x_offset = int(x_middle - (cube.shape[0] / 2))
            y_offset = int(y_middle - (cube.shape[1] / 2))

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

        cv2.rectangle(final, (self.green_area[0], self.green_area[2]), (self.green_area[3], self.green_area[1]),
                      (0, 255, 0), 2)
        cv2.imwrite('new_overlay.png', final)

        w, h, _ = self.background.shape
        final = cv2.resize(final, (int(h / Divide_Factor), int(w / Divide_Factor)))
        return final


class Img_generator:

    def __init__(self, green_area=None, num=3):
        self.original_num = num
        self.background = cv2.imread('background.jpeg')
        self.cubes = [cv2.imread('final_cube1.jpeg'), cv2.imread('final_cube2.png'), cv2.imread('final_cube3.png')]

        # the area where we want the cubes to be
        # format: [x_min, x_max, y_min, y_max]
        if green_area is None:
            self.green_area = [0, 320, 0, 215]
        else:
            self.green_area = green_area
        self.in_green_counter = 0

        # do we have a cube in the gripper?
        self.in_hand = False

        # the bounds of the board
        self.min_ = (40, 40)
        self.max_ = (280, 390)
        self.GRAB, self.DROP = range(2)

        # generate the cubes locations and image
        self.locations = self.generate_random(num)
        self.image = self.generate_img()

    # returns img, if the grab was successful/drop is in green area and if all cubes are in the green area
    def go_to_location(self, location) -> (np.ndarray, (int, bool), bool):
        # if putting down a cube
        if self.in_hand:
            self.locations.append(location)
            self.in_hand = False
            in_green = self.is_in_green(location)
            if in_green:
                self.in_green_counter += 1

            self.image = self.generate_img()
            return self.image, (self.DROP, in_green), (self.in_green_counter == self.original_num)

        # if trying to grab a cube
        for loc in self.locations:
            if distance(loc, location) < EPSILON:
                # remove cube
                self.locations.remove(loc)
                self.image = self.generate_img()
                self.in_hand = True
                if self.is_in_green(loc):
                    self.in_green_counter -= 1
                    return self.image, (self.GRAB, -1), False

                return self.image, (self.GRAB, True), False

        cv2.imwrite("current.png", self.image)

        # if grab was unsuccessful
        return self.image, (self.GRAB, False), False

    # num: number of cubes to be generated *optional*
    def reset(self, num=0):
        if num:
            self.original_num = num

        self.in_green_counter = 0
        self.locations = self.generate_random(self.original_num)
        self.image = self.generate_img()
        return self.image

    def get_img(self):
        return self.image

    def action_bounds(self):
        return (self.min_[0], self.max_[0]), (self.min_[1],self.max_[1])

    def observation_bounds(self):
        return (0, self.background.shape[0]), (0, self.background.shape[1])

    def generate_random(self, num):
        min_, max_ = self.min_, self.max_
        min_distance = 50
        locs = []
        # check if area is only the right side of the board
        is_area_right_size = self.green_area[0] == 0 and self.green_area[2] == 0

        while num > 0:
            # generate random location for right side area
            if is_area_right_size:
                new_loc = (randint(self.green_area[3] + Caution_Factor, max_[1] - Caution_Factor), randint(min_[0] + Caution_Factor, max_[0] - Caution_Factor))
            # generate random location for any other area (slower)
            else:
                new_loc = (randint(min_[1], max_[1]), randint(min_[0], max_[0]))

                # we don't want the starting position to be in the green area
                if self.is_in_green(new_loc):
                    continue

            # check if the new location is too close to the previous ones
            fine = True
            for l in locs:
                if distance(new_loc, l) < min_distance:
                    fine = False

            if fine:
                locs.append(new_loc)
                num -= 1

        return locs

    def is_in_green(self, location):
        x, y = location
        return self.green_area[0] < y < self.green_area[1] and self.green_area[2] < x < self.green_area[3]

    def generate_img(self):
        background = self.background
        cubes = self.cubes
        locations = self.locations
        final = background

        new_cube = np.zeros(background.shape)
        for loc in locations:
            cube = cubes[randint(0, len(cubes) - 1)]
            x_middle, y_middle = loc

            x_offset = int(x_middle - (cube.shape[0] / 2))
            y_offset = int(y_middle - (cube.shape[1] / 2))

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

        cv2.rectangle(final, (self.green_area[0], self.green_area[2]), (self.green_area[3], self.green_area[1]),
                      (0, 255, 0), 2)
        cv2.imwrite('new_overlay.png', final)

        w, h, _ = self.background.shape
        final = cv2.resize(final, (int(h / Divide_Factor), int(w / Divide_Factor)))
        return final


if __name__ == '__main__':
    i = Img_generator([0, 320, 0, 215], num=3)
    cv2.imwrite('temp1.png', i.get_img())
    res = i.go_to_location((i.locations[1][0] + 4, i.locations[1][1]))
    res = i.go_to_location((50, 50))

    cv2.imwrite('temp2.png', res[0])
