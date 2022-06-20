import math
from typing import Tuple
import cv2
# simulator and environment parameters

cf = 50  # caution factor - how many pixels from each edge in the photo should be kept without cubes
epsilon_function = lambda h, w: (h + w) / 50  # used in determination of reward, has distance squared units
step_size_function = lambda h, w: ((h + w) / 300, (h + w) / 300)  # size of steps that can be taken
max_unsuccessful_grabs = 4000  # maximum number of grabs before we declare the episode unsuccessful

# reinforcement learning-specific parameters

training_start = 500_000  # number of random steps that are taken before online training starts
learning_rate_max_rate = 1e-3  # learning rate varies linearly between the minimum and maximum rates here
learning_rate_min_rate = 1e-6
buffer_size = 200_000  # the buffer size for the algorithm

# image creator parameters
background = cv2.imread('rl_project/images/background.png')
cubes = list(map(cv2.imread, ['rl_project/images/final_cube1.png', 'rl_project/images/final_cube2.png',
                                      'rl_project/images/final_cube3.png']))
resize_factor = 4  # factor by which images created are made smaller (decreasing the amount of data)

# multi-cube simulator parameters

min_distance_between_cubes = 50  # will not place two simulated cubes closer than this distance
capture_bonus_factor = 0.8  # controls the relative amount of reward given for cubes already captured
                            # versus the new cube to be captured

# predictor parameters

base_model_path = "/tmp/pycharm_project_970/rl_project/supervised"

# debug settings

debug = True  # enables various additional prints and writes for debugging purposes
reporting_frequency = 10000

print_steps = False
announce_out_of_bounds = False
print_state = True
use_real_pos = False
