from random import randint
import numpy as np
from rl_project.config import *
from rl_project.image_creator import create_multiimage
from rl_project.widowx import GenericWidowX


def distance(a, b):
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)) ** 0.5


# noinspection DuplicatedCode
class WidowXMultiSimulator(GenericWidowX):

    # INTERFACE FUNCTIONS

    def __init__(self, num=3):
        self.original_num = num
        self.min_bounds = (cf, cf)
        self.max_bounds = (background.shape[0] - cf, background.shape[1] - cf)
        self.h, self.w = background.shape[0], background.shape[1]
        self.found = False
        self.locations = self.generate_random(num)
        self.image = create_multiimage(self.locations)
        self.pos = ((self.min_bounds[0] + self.max_bounds[1]) / 2, (self.min_bounds[0] + self.max_bounds[1]) / 2)

    def step(self, steps: Tuple[float, float]) -> Tuple[float, float]:
        step_size_x, step_size_y = step_size_function(self.h, self.w)
        step_x, step_y = steps[0] * step_size_x, steps[1] * step_size_y
        bound_x_min, bound_x_max = self.min_bounds
        bound_y_min, bound_y_max = self.max_bounds
        change = (step_x, step_y)
        if announce_out_of_bounds and not (bound_x_min <= self.pos[0] + change[0] <= bound_x_max):
            print("X change out of bounds!")
        if announce_out_of_bounds and not (bound_y_min <= self.pos[1] + change[1] <= bound_y_max):
            print("Y change out of bounds!")
        self.pos = (WidowXMultiSimulator.clip(self.pos[0] + change[0], bound_x_min, bound_x_max),
                    WidowXMultiSimulator.clip(self.pos[1] + change[1], bound_y_min, bound_y_max))
        if print_steps:
            print("Step: ", change)
        for loc in self.locations:
            if distance(loc, self.pos) < epsilon_function(self.max_bounds[0], self.min_bounds[0]):
                # remove cube
                self.locations.remove(loc)
                self.image = create_multiimage(self.locations)
        if debug:
            cv2.imwrite("current.png", self.image)
        return self.pos

    def reset(self):
        self.locations = self.generate_random(self.original_num)
        self.image = create_multiimage(self.locations)
        return self.image

    def get_image(self):
        return self.image

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (cf, self.w - cf), (cf, self.h - cf)

    def get_pos(self) -> Tuple[float, float]:
        return self.pos

    def max_reward(self):
        return (self.original_num * capture_bonus_factor + 1) * epsilon_function(self.h, self.w) / self.diag_length_sq()

    def eval_pos(self) -> Tuple[int, float]:
        reward = (epsilon_function(self.h, self.w) - self.max_distance_sq_from_target()) / self.diag_length_sq()
        capture_reward = (self.original_num - len(self.locations)) \
                         * capture_bonus_factor * epsilon_function(self.h, self.w) / self.diag_length_sq()
        self.found = reward >= (self.max_reward() * 0.5)
        return self.original_num - len(self.locations) + (1 if self.found else 0), reward

    # HELPER FUNCTIONS

    def max_distance_sq_from_target(self) -> float:
        max_distance = 0
        for loc in self.locations:
            max_distance = max(max_distance, distance(loc, self.pos))
        return max_distance

    def generate_random(self, num):
        min_, max_ = self.min_bounds, self.max_bounds
        locs = []
        while num > 0:
            new_loc = (randint(min_[1], max_[1]), randint(min_[0], max_[0]))

            # check if the new location is too close to the previous ones
            fine = True
            for l in locs:
                if distance(new_loc, l) < min_distance_between_cubes:
                    fine = False

            if fine:
                locs.append(new_loc)
                num -= 1

        return locs
