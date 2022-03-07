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

    def normalized_to_regular(self, x, y, distance=None):
        return ((1 - x) * (self.widowx.bounds()[0][0] - self.widowx.bounds()[0][1]) + x * (self.widowx.bounds()[0][1]),
                (1 - y) * (self.widowx.bounds()[1][0] - self.widowx.bounds()[1][1]) + y * (self.widowx.bounds()[1][1]),
                distance * self.widowx.diag_length_sq() if distance else None)

    def regular_to_normalized(self, x, y, distance=None):
        return ((x - self.widowx.bounds()[0][0]) / (self.widowx.bounds()[0][1] - self.widowx.bounds()[0][0]),
                (y - self.widowx.bounds()[1][0]) / (self.widowx.bounds()[1][1] - self.widowx.bounds()[1][0]),
                distance / self.widowx.diag_length_sq() if distance else None)

    def __init__(self, widowx: WidowXSimulator = WidowXSimulator(None)):  # later remove type annotation
        self.widowx = widowx
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.successful_grabs = 0
        self.iteration = 0
        self.unsuccessful = 0
        self.np_random = None
        self.reset()

    def step(self, action) -> Tuple[ndarray, float, bool, dict]:
        if not isinstance(action, int):
            action = action[0]
        # assert self.action_space.contains(action), f"Action {action} not in action space, {self.action_space}"

        if debug:
            print("calling step: ", action, " x_cube: ", self.widowx.x_cube, " y_cube: ", self.widowx.y_cube)
        self.iteration += 1

        if self.iteration == training_start:
            global count_training_rounds
            count_training_rounds = True

        self.widowx.step(action)
        is_cube_in_gripper, reward = self.widowx.eval_pos()
        new_state = np.array(self.regular_to_normalized(self.widowx.pos[0], self.widowx.pos[1],
                                                        self.widowx.distance_sq_from_target()), dtype=np.float32)

        if is_cube_in_gripper:
            self.successful_grabs += 1
        else:
            self.unsuccessful += 1
            if self.unsuccessful == 75:
                self.unsuccessful = 0
                is_cube_in_gripper = True
        if debug:
            print(f'got reward: {reward}. iteration: {self.iteration}, successful grab: {self.successful_grabs}')
            print(f"state: {new_state, is_cube_in_gripper}")
        return new_state, reward, is_cube_in_gripper, {}

    def reset(self) -> ndarray:
        if debug:
            print("calling reset in env")
        self.widowx.reset()
        self.unsuccessful = 0
        new_state = np.array(self.regular_to_normalized(self.widowx.pos[0], self.widowx.pos[1],
                                                        self.widowx.distance_sq_from_target()), dtype=np.float32)
        return new_state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed


def register_env():
    gym.envs.register(
        id='SimulatorEnv-v0',
        entry_point='simulator_env:SimulatorEnv',
        max_episode_steps=1000,
    )
