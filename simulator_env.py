import math
from typing import Tuple

import gym
import numpy as np
from gym.spaces import Box
from gym.utils import seeding
from numpy import ndarray
from config import *
from widowx_simulator import WidowXSimulator

actions_dim = 1


class SimulatorEnv(gym.Env):
    def render(self, mode='human'):
        pass

    def normalized_to_regular(self, x, y, distance=None, angle=None):
        return ((1 - x) * (self.widowx.bounds()[0][0] - self.widowx.bounds()[0][1]) + x * (self.widowx.bounds()[0][1]),
                (1 - y) * (self.widowx.bounds()[1][0] - self.widowx.bounds()[1][1]) + y * (self.widowx.bounds()[1][1]),
                distance * self.widowx.diag_length_sq() if distance else 0.0,
                2 * math.pi * angle - math.pi if angle else 0.0)

    def regular_to_normalized(self, x, y, distance=None, angle=None):
        return ((x - self.widowx.bounds()[0][0]) / (self.widowx.bounds()[0][1] - self.widowx.bounds()[0][0]),
                (y - self.widowx.bounds()[1][0]) / (self.widowx.bounds()[1][1] - self.widowx.bounds()[1][0]),
                distance / self.widowx.diag_length_sq() if distance else 0.0,
                angle / (2 * math.pi) + 0.5 if angle else 0.0)

    def __init__(self, widowx: WidowXSimulator = WidowXSimulator(None)):  # later remove type annotation
        self.widowx = widowx
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.successful_grabs = 0
        self.iteration = 0
        self.unsuccessful = 0
        self.reset()

    @staticmethod
    def get_direction_between_points(a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def step(self, action) -> Tuple[ndarray, float, bool, dict]:
        # assert self.action_space.contains(action), f"Action {action} not in action space, {self.action_space}"

        if debug and reporting_frequency > 0 and self.iteration % reporting_frequency == 0:
            print("calling step: ", action, " x_cube: ", self.widowx.x_cube, " y_cube: ", self.widowx.y_cube)
        self.iteration += 1

        if self.iteration == training_start:
            global count_training_rounds
            count_training_rounds = True

        self.widowx.step(action)
        is_cube_in_gripper, reward = self.widowx.eval_pos()
        state_raw = self.regular_to_normalized(self.widowx.pos[0], self.widowx.pos[1],
                                                        self.widowx.distance_sq_from_target(),
                                                        SimulatorEnv.get_direction_between_points(self.widowx.pos,
                                                                                                  (self.widowx.x_cube,
                                                                                                   self.widowx.y_cube)))
        new_state = np.array(state_raw, dtype=np.float32)
        if len(list(filter(lambda x: math.isnan(x), state_raw))) > 0:
            print("nan in state")
        if is_cube_in_gripper:
            self.successful_grabs += 1
        else:
            self.unsuccessful += 1
            if self.unsuccessful == max_unsuccessful_grabs:
                self.unsuccessful = 0
                is_cube_in_gripper = True
        if debug and reporting_frequency > 0 and self.iteration % reporting_frequency == 0:
            print(f'got reward: {reward}. iteration: {self.iteration}, successful grab: {self.successful_grabs}')
            print(f"state: {new_state, is_cube_in_gripper}")
            if self.successful_grabs > 0:
                print(f"score (lower is better): {self.iteration / (1000 * self.successful_grabs)}")
            if self.iteration < training_start:
                print("training round (random steps)")
            else:
                print("actor round (real steps)")
        return new_state, reward, True if is_cube_in_gripper else False, {}  # the `if` is not useless

    def reset(self) -> ndarray:
        if debug and reporting_frequency > 0 and self.iteration % reporting_frequency == 0:
            print("calling reset in env")
        self.widowx.reset()
        self.unsuccessful = 0
        state_raw = self.regular_to_normalized(self.widowx.pos[0], self.widowx.pos[1],
                                               self.widowx.distance_sq_from_target(),
                                               SimulatorEnv.get_direction_between_points(self.widowx.pos,
                                                                                         (self.widowx.x_cube,
                                                                                          self.widowx.y_cube)))
        new_state = np.array(state_raw, dtype=np.float32)
        if len(list(filter(lambda x: math.isnan(x), state_raw))) > 0:
            print("nan in state")
        return new_state


def register_env():
    gym.envs.register(
        id='SimulatorEnv-v0',
        entry_point='simulator_env:SimulatorEnv',
        max_episode_steps=1000,
    )
