import random
from typing import Tuple

import cv2

from config import *

count_training_rounds = False
rewards = []  # for debug


class WidowXSimulator:
    def __init__(self, widowx):
        self.widowx = widowx
        self.background = cv2.imread('../background.jpeg')
        self.w, self.h, _ = (430, 320, 3)  # self.background.shape
        self.x_cube = random.randint(cf, self.w - cf)
        self.y_cube = random.randint(cf, self.h - cf)
        self.pos = ((self.bounds()[0][0] + self.bounds()[0][1]) / 2, (self.bounds()[1][0] + self.bounds()[1][1]) / 2)
        self.found = False
        self.training_rounds = 0
        self.repetitions = 0
        self.last_action = 0
        if debug:
            print(f"Size: {self.w, self.h}")

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (cf, self.w - cf), (cf, self.h - cf)

    @staticmethod
    def clip(value: float, min_value: float, max_value: float) -> float:
        return max(min(value, max_value), min_value)

    def step(self, steps: Tuple[float, float]) -> Tuple[float, float]:
        step_size_x, step_size_y = step_sizes
        step_x, step_y = steps[0] * step_size_x, steps[1] * step_size_y
        bound_x_min, bound_x_max = self.bounds()[0]
        bound_y_min, bound_y_max = self.bounds()[1]
        change = (step_x, step_y)
        if not (bound_x_min <= self.pos[0] + change[0] <= bound_x_max):
            if debug:
                print("X change out of bounds!")
        if not (bound_y_min <= self.pos[1] + change[1] <= bound_y_max):
            if debug:
                print("Y change out of bounds!")
        self.pos = (WidowXSimulator.clip(self.pos[0] + change[0], bound_x_min, bound_x_max),
                    WidowXSimulator.clip(self.pos[1] + change[1], bound_y_min, bound_y_max))
        self.last_action = steps
        if debug:
            print("Step: ", change)
        return self.pos

    def get_pos(self) -> Tuple[float, float]:
        return self.pos

    def diag_length_sq(self) -> float:
        return (self.bounds()[0][1] - self.bounds()[0][0]) ** 2 + (self.bounds()[1][1] - self.bounds()[1][0]) ** 2

    def distance_sq_from_target(self) -> float:
        return (self.pos[0] - self.x_cube) ** 2 + (self.pos[1] - self.y_cube) ** 2

    def eval_pos(self) -> Tuple[bool, float]:
        reward = (epsilon - self.distance_sq_from_target()) / self.diag_length_sq()
        self.found = reward >= 0
        return self.found, reward

    def reset(self):
        self.x_cube = random.randint(cf, self.w - cf)
        self.y_cube = random.randint(cf, self.h - cf)
        self.found = False
        self.pos = ((self.bounds()[0][0] + self.bounds()[0][1]) / 2, (self.bounds()[1][0] + self.bounds()[1][1]) / 2)
        # self.repetitions = 0
        if debug:
            print(f"calling reset in simulator")