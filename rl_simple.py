import math
import random

import gym
import numpy as np
from gym.spaces import Box
from stable_baselines3 import TD3

STEPS_PER_EPISODE = 400
EPSILON = 0.1


class BasicEnv(gym.Env):
    def __init__(self):
        # define action space
        # (step size, step direction)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # (my_x, my_y, wanted_x, wanted_y)
        self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # generate location
        self.wanted_location = self.generate_location()
        self.current_location = self.generate_location()

        # count successful episodes
        self.success_count = 0

        # count steps
        self.iteration = 0

    @staticmethod
    def generate_location():
        x = round(random.random(), 2)
        y = round(random.random(), 2)
        return x, y

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        return round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 2)

    @staticmethod
    def in_bounds(x):
        return round(max(0, min(1, x)),2)

    def print(self, step, reward=None, reset=False):
        action = f'___Step: {[round(x, 2) for x in step]}, ' if not reset else '___Reset: '
        reward = f', reward: {reward} ' if reward is not None else ''
        print(f'{action} {self.current_location} -> {self.wanted_location}')
        print(f'Distance: {self.get_distance(*self.current_location, *self.wanted_location)} {reward}')
        print(f'iteration: {self.iteration}, Success: {self.success_count}')

    def step(self, action):
        x, y = self.current_location
        # take step
        if action[1] >= 0:
            x += action[0] / 10
        else:
            y += action[0] / 10

        # make sure we stay in bounds
        x = self.in_bounds(x)
        y = self.in_bounds(y)
        self.current_location = x, y

        # check if we reached the wanted location
        distance = self.get_distance(x, y, self.wanted_location[0], self.wanted_location[1])
        reward = round((math.sqrt(2) - distance) / math.sqrt(2), 2)
        self.print(reward=reward, step=action)
        if distance < EPSILON:
            self.success_count += 1
            return (x, y, self.wanted_location[0], self.wanted_location[1]), 1, True, {}

        # return new state
        self.iteration += 1
        done = self.iteration >= STEPS_PER_EPISODE
        return (x, y, self.wanted_location[0], self.wanted_location[1]), reward, done, {}

    def reset(self):
        self.current_location = self.generate_location()
        self.wanted_location = self.generate_location()
        self.iteration = 0
        self.print(step=None, reset=True)
        return self.current_location[0], self.current_location[1], self.wanted_location[0], self.wanted_location[1]


def start_rl():
    env = BasicEnv()

    model = TD3("MlpPolicy", env, buffer_size=100000, learning_starts=100, train_freq=(5, "episode"),
                tensorboard_log="tb_log_3")
    model.learn(total_timesteps=100000)
    model.save("rl_simple")


if __name__ == '__main__':
    start_rl()
