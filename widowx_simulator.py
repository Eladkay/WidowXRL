import random
from rl_project.image_creator import *

from rl_project.config import *
from rl_project.widowx import GenericWidowX


class WidowXSimulator(GenericWidowX):
    def __init__(self):
        self.w, self.h, _ = background.shape
        self.x_cube = random.randint(cf, self.w - cf)
        self.y_cube = random.randint(cf, self.h - cf)
        self.pos = ((self.bounds()[0][0] + self.bounds()[0][1]) / 2, (self.bounds()[1][0] + self.bounds()[1][1]) / 2)
        self.found = False
        self.image = create_binary_img(self.y_cube, self.x_cube)
        if debug:
            print(f"Size: {self.w, self.h}")

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (cf, self.w - cf), (cf, self.h - cf)

    def step(self, steps: Tuple[float, float]) -> Tuple[float, float]:
        step_size_x, step_size_y = step_size_function(self.h, self.w)
        step_x, step_y = steps[0] * step_size_x, steps[1] * step_size_y
        bound_x_min, bound_x_max = self.bounds()[0]
        bound_y_min, bound_y_max = self.bounds()[1]
        change = (step_x, step_y)
        if announce_out_of_bounds and not (bound_x_min <= self.pos[0] + change[0] <= bound_x_max):
            print("X change out of bounds!")
        if announce_out_of_bounds and not (bound_y_min <= self.pos[1] + change[1] <= bound_y_max):
            print("Y change out of bounds!")
        self.pos = (GenericWidowX.clip(self.pos[0] + change[0], bound_x_min, bound_x_max),
                    GenericWidowX.clip(self.pos[1] + change[1], bound_y_min, bound_y_max))
        if print_steps:
            print("Step: ", change)
        return self.pos

    def get_pos(self) -> Tuple[float, float]:
        return self.pos

    def distance_sq_from_target(self) -> float:
        return (self.pos[0] - self.x_cube) ** 2 + (self.pos[1] - self.y_cube) ** 2

    def eval_pos(self) -> Tuple[int, float]:
        reward = (epsilon_function(self.h, self.w) - self.distance_sq_from_target()) / self.diag_length_sq()
        self.found = reward >= 0
        return 1 if self.found else 0, reward

    def reset(self):
        self.x_cube = random.randint(cf, self.w - cf)
        self.y_cube = random.randint(cf, self.h - cf)
        self.found = False
        self.pos = ((self.bounds()[0][0] + self.bounds()[0][1]) / 2, (self.bounds()[1][0] + self.bounds()[1][1]) / 2)
        self.image = create_binary_img(self.y_cube, self.x_cube)

    def get_image(self):
        return self.image
