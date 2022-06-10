from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from rl_project.widowx_simulator import *
from rl_project.simulator_env import *
from rl_project.config import *


def check_simulator_env():
    simulator = WidowXMultiSimulator()
    env = SimulatorEnv(simulator)  # also possible: gym.make('SimulatorEnv-v0', widowx=simulator)
    check_env(env, warn=True, skip_render_check=True)


def do_agent():
    simulator = WidowXMultiSimulator()
    env = SimulatorEnv(simulator)  # also possible: gym.make('SimulatorEnv-v0', widowx=simulator)
    model = TD3('MultiInputPolicy', env, verbose=1, learning_starts=training_start, train_freq=(10, "episode"),
                learning_rate=(lambda x: x * (learning_rate_max_rate - learning_rate_min_rate)
                                         + learning_rate_min_rate), tensorboard_log="tb_logs", buffer_size=buffer_size)
    model.learn(total_timesteps=1000000, log_interval=100, eval_freq=1000, n_eval_episodes=5,
                eval_log_path="./rl_logs/")
    model.save("td3_simulator")


def start_rl():
    register_env()
    check_simulator_env()
    do_agent()



if __name__ == "__main__":
    start_rl()
