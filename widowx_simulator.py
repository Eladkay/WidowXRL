import math
import random
from typing import Tuple

import cv2

from config import *

count_training_rounds = False
rewards = []  # for debug
actions = []  # for debug


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

    def step(self, direction: float) -> Tuple[float, float]:
        step_size_x, step_size_y = step_sizes
        bound_x_min, bound_x_max = self.bounds()[0]
        bound_y_min, bound_y_max = self.bounds()[1]
        change = (step_size_x * math.cos(math.pi * direction), step_size_y * math.sin(math.pi * direction))
        if not (bound_x_min <= self.pos[0] + change[0] <= bound_x_max):
            change = (0, change[1])  # design decision - if exceeds, zero it out (and not do partial steps)
            if debug:
                print("X change out of bounds!")
        if not (bound_y_min <= self.pos[1] + change[1] <= bound_y_max):
            change = (change[0], 0)
            if debug:
                print("Y change out of bounds!")
        self.pos = (self.pos[0] + change[0], self.pos[1] + change[1])
        if count_training_rounds:
            self.training_rounds += 1
        if abs(self.last_action - direction) < delta:
            self.repetitions += 1
        else:
            self.repetitions = 0
        self.last_action = direction
        if debug:
            print("Step: ", change)
        histogram[int(direction * 5 + 5)] += 1
        actions.append(direction)
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
        if count_training_rounds and self.training_rounds < first_rounds:
            reward += first_rounds_bonus / self.diag_length_sq()
        if 0 < penalize_repetitions < self.repetitions:
            reward += (self.repetitions - penalize_repetitions) * repetition_penalty / self.diag_length_sq()
        rewards.append(reward)
        return self.found, reward

    def reset(self):
        self.x_cube = random.randint(cf, self.w - cf)
        self.y_cube = random.randint(cf, self.h - cf)
        self.found = False
        self.pos = ((self.bounds()[0][0] + self.bounds()[0][1]) / 2, (self.bounds()[1][0] + self.bounds()[1][1]) / 2)
        # self.repetitions = 0
        if debug:
            print(f"calling reset in simulator")
