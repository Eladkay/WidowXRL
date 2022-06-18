import math
from typing import Tuple

import gym
import tensorboard
import tensorflow
import numpy as np
from gym.spaces import Box, Dict
from gym.utils import seeding
from numpy import ndarray
from rl_project.config import *
from rl_project.widowx import GenericWidowX
from rl_project.widowx_multisimulator import WidowXMultiSimulator
from rl_project.widowx_simulator import WidowXSimulator

actions_dim = 1


class SimulatorEnv(gym.Env):

    def regular_to_normalized(self, x, y, distance=None, angle=None):
        """
        Normalizes values to be in their standard range: [0, 1] for x, y and distance which are given in absolute
        units and [0, 1] for angles given in the range (-pi, pi]
        :param x: The x value to normalize
        :param y: The y value to normalize
        :param distance: The distance to normalize, if any
        :param angle: The angle to normalize, if any
        :return: The normalized values. If distance or angle are omitted, 0.0 is returned in their place.
        """
        return ((x - self.widowx.bounds()[0][0]) / (self.widowx.bounds()[0][1] - self.widowx.bounds()[0][0]),
                (y - self.widowx.bounds()[1][0]) / (self.widowx.bounds()[1][1] - self.widowx.bounds()[1][0]),
                distance / self.widowx.diag_length_sq() if distance else 0.0,
                angle / (2 * math.pi) + 0.5 if angle else 0.0)
        return clip(x_n, 0, 1), clip(y_n, 0, 1), clip(d_n, 0, 1), clip(a_n, 0, 1)

    def normalized_to_regular(self, x, y, distance=None, angle=None):
        """
        Performs the inverse function to SimulatorEnv#regular_to_normalized.
        :return: x, y and distance values given in absolute units and angle values given in the range (-pi, pi].
        """
        return ((1 - x) * (self.widowx.bounds()[0][0] - self.widowx.bounds()[0][1]) + x * (self.widowx.bounds()[0][1]),
                (1 - y) * (self.widowx.bounds()[1][0] - self.widowx.bounds()[1][1]) + y * (self.widowx.bounds()[1][1]),
                distance * self.widowx.diag_length_sq() if distance else 0.0,
                2 * math.pi * angle - math.pi if angle else 0.0)

    @staticmethod
    def reduce_dim(old_img: ndarray) -> ndarray:
        """
        Reduces the dimension of the image from 3 to 2 by performing a weighted average between the (less important)
        blue component and the other components. This is done because the cubes that are used are red, and there is no
        natural blue in the images.
        :param old_img: The image whose dimension to reduce.
        :return: The reduced-dimension image.
        """
        h, w, d = old_img.shape
        assert d == 3
        new_img = np.zeros((h, w, 2))
        new_img[:, :, 0] = old_img[:, :, 0] * 2 / 3 + old_img[:, :, 2] / 3
        new_img[:, :, 1] = old_img[:, :, 1] * 2 / 3 + old_img[:, :, 2] / 3
        return new_img

    def __init__(self, widowx: GenericWidowX):
        self.widowx = widowx

        # Action spaces supported by GenericWidowX are given as [-1, 1] multiplies of the step size.
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                
        # Environment varies between experiments - set by default to be a dict of position and image.
        self.observation_space = Dict({"pos": Box(low=0, high=1, shape=(2, ), dtype=np.float32),
                                       "image": Box(low=0, high=255, shape=widowx.get_image_shape(), dtype=np.uint8)})
        self.successful_grabs = 0
        self.iteration = 0
        self.unsuccessful = 0
        self.failures = 0
        self.prediction = None
        self.reset()

    @staticmethod
    def get_direction_between_points(a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def get_state(self) -> dict:
        self_pos_normalized = self.regular_to_normalized(self.widowx.pos[0], self.widowx.pos[1])
        obj_pos_guess = (self.widowx.x_cube, self.widowx.y_cube) \
            if use_real_pos else self.prediction
        obj_pos_normalized = self.regular_to_normalized(obj_pos_guess[0], obj_pos_guess[1])
        new_state = {"self_pos": np.array([self_pos_normalized[0], self_pos_normalized[1]]),
                     "obj_pos": np.array([obj_pos_normalized[0], obj_pos_normalized[1]])}
        return new_state

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        if debug and reporting_frequency > 0 and self.iteration % reporting_frequency == 0:
            if self.widowx is WidowXSimulator:
                print("calling step: ", action, " x_cube: ", self.widowx.x_cube, " y_cube: ", self.widowx.y_cube,
                      "x_actor: ", self.widowx.get_pos()[0], " y_actor: ", self.widowx.get_pos()[1])
            else:
                print("calling step: ", action, " furthest distance", self.widowx.max_distance_sq_from_target(),
                      "x_actor: ", self.widowx.get_pos()[0], " y_actor: ", self.widowx.get_pos()[1])
                print(self.widowx.locations)
        self.iteration += 1

        if self.iteration == training_start:
            global count_training_rounds
            count_training_rounds = True

        self.widowx.step(action)
        is_cube_in_gripper, reward = self.widowx.eval_pos()
        pos_normalized = self.regular_to_normalized(self.widowx.get_pos()[0], self.widowx.get_pos()[1])
        new_state = {"pos": np.array([pos_normalized[0], pos_normalized[1]]),
                "image": self.widowx.get_image()}
        if is_cube_in_gripper:
            self.successful_grabs += 1
        else:
            self.unsuccessful += 1
            if self.unsuccessful == max_unsuccessful_grabs:
                self.unsuccessful = 0
                is_cube_in_gripper = True
                if self.iteration > training_start:
                    self.failures += 1
        if debug and reporting_frequency > 0 and self.iteration % reporting_frequency == 0:
            print(f'got reward: {reward}. iteration: {self.iteration}, successful grab: {self.successful_grabs}, '
                  f'failures after training start: {self.failures}')
            if print_state:
                print(f"state: {new_state, is_cube_in_gripper}")
            if self.successful_grabs > 0:
                print(f"score (lower is better): {self.iteration / (1000 * self.successful_grabs)}")
            if self.iteration < training_start:
                print("training round (random steps)")
            else:
                print("actor round (real steps)")
        return new_state, reward, True if is_cube_in_gripper else False, {}  # the `if` is not useless

    def reset(self) -> dict:
        if debug and reporting_frequency > 0 and self.iteration % reporting_frequency == 0:
            print("calling reset in env")
        self.widowx.reset()
        self.unsuccessful = 0
        pos_normalized = self.regular_to_normalized(self.widowx.get_pos()[0], self.widowx.get_pos()[1])
        return {"pos": np.array([pos_normalized[0], pos_normalized[1]]),
                "image": self.widowx.get_image()}


def register_env():
    gym.envs.register(
        id='SimulatorEnv-v0',
        entry_point='rl_project.simulator_env:SimulatorEnv',
        max_episode_steps=1000,
    )

