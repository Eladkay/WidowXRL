import math
from typing import Tuple
import cv2

class LearningRateFunctions:
    @staticmethod
    def constant(x: float) -> float:
        return 1.0

    @staticmethod
    def linear(x: float) -> float:
        return x

    one_over_one_minus_e = 1 / (1 - math.e)

    @staticmethod
    def exponential(x: float) -> float:
        return (1 - (math.e ** x)) * LearningRateFunctions.one_over_one_minus_e

    one_over_one_minus_cos_of_one = 1 / (1 - math.cos(1))

    @staticmethod
    def cosine(x: float) -> float:
        return (1-math.cos(x)) * LearningRateFunctions.one_over_one_minus_cos_of_one

    @staticmethod
    def quadratic(x: float) -> float:
        return x ** 2

    @staticmethod
    def inv_quadratic(x: float) -> float:
        return 1 - (x - 1) ** 2


class StepSizeFunctions:
    @staticmethod
    def constant(h: float, w: float) -> Tuple[float, float]:
        return 2.5, 2.5

    @staticmethod
    def linear_avg(h: float, w: float) -> Tuple[float, float]:
        return (h + w) / 300, (h + w) / 300


class EpsilonFunctions:
    @staticmethod
    def constant(h: float, w: float) -> float:
        return 15

    @staticmethod
    def linear_avg(h: float, w: float) -> float:
        return (h + w) / 50


# simulator and environment parameters

cf = 50  # caution factor
epsilon_function = EpsilonFunctions.linear_avg  # distance squared units
step_size_function = StepSizeFunctions.linear_avg  # distance units
max_unsuccessful_grabs = 4000

# algorithm parameters

training_start = 500_000
learning_rate_function = LearningRateFunctions.inv_quadratic
learning_rate_max_rate = 1e-3
learning_rate_min_rate = 1e-6
buffer_size = 200_000

# image creator parameters
background = cv2.imread('rl_project/images/background.png')
cubes = [cv2.imread('rl_project/images/final_cube1.png'), cv2.imread('rl_project/images/final_cube2.png'), cv2.imread(
    'rl_project/images/final_cube3.png')]
write_image_to_file = False
resize_factor = 4

# predictor parameters

base_model_path = "/tmp/pycharm_project_970/rl_project/supervised"

# multisimulator parameters

min_distance_between_cubes = 50
capture_bonus_factor = 0.8

# debug settings

debug = True
reporting_frequency = 10000
print_steps = False
announce_out_of_bounds = False
print_state = False
