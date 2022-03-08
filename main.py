from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from widowx_simulator import *
from simulator_env import *
from config import *


def check_simulator_env():
    simulator = WidowXSimulator(None)
    env = gym.make('SimulatorEnv-v0', widowx=simulator)
    check_env(env, warn=True, skip_render_check=True)


def do_agent():
    env = gym.make('SimulatorEnv-v0')
    model = TD3('MultiInputPolicy', env, verbose=1, learning_starts=training_start, train_freq=(10, "episode"),
                learning_rate=(lambda x: learning_rate_function(x) * (learning_rate_max_rate - learning_rate_min_rate)
                                         + learning_rate_min_rate), tensorboard_log="tb_logs", buffer_size=buffer_size)
    model.learn(total_timesteps=10000000, log_interval=100, eval_freq=1000, n_eval_episodes=5,
                eval_log_path="./rl_logs/")
    model.save("td3_simulator")


def start_rl():
    register_env()
    check_simulator_env()
    do_agent()

    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # # model = TD3("MlpPolicy", env, buffer_size=500_000, learning_starts=training_start, train_freq=(1_000, "step"),
    # #             gradient_steps=5, action_noise=action_noise, verbose=1)
    # model = TD3("MlpPolicy", "SimulatorEnv-v0", action_noise=action_noise, verbose=1, learning_starts=training_start,
    #             train_freq=(1_000, "step"), gradient_steps=3,
    #             learning_rate=(lambda x: learning_rate_base_rate * learning_rate_function(x)))
    # model.learn(total_timesteps=277_000, eval_freq=1000, n_eval_episodes=5, eval_log_path="./rl_logs/",
    #             log_interval=10)


def start_random():
    simulator = WidowXSimulator(None)
    i = 0
    for _ in range(10000):
        i += 1
        step = random.randint(0, 15)
        simulator.step(step)
        print("Step: ", step)
        print(f"Pos: {simulator.get_pos()}")
        print(f"Eval_pos: {simulator.eval_pos()}")
        print(f"Distance: {simulator.distance_sq_from_target()}")
        print(f"Iteration: {i}")
        print("")
        if simulator.found:
            break
    print(f"Size: {simulator.w, simulator.h}")


if __name__ == "__main__":
    start_rl()
