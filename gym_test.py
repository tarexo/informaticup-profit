import gym
from gym.envs.registration import register
from gym.utils.env_checker import check_env

from helper.constants.settings import *


def register_env(name):
    register(id="Profit-v0", entry_point="profit_gym:ProfitGym", max_episode_steps=300)


def run_gym(env, seed=None):
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed)

    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if truncated:
            env.reset()
        else:
            env.render()

    env.close()


if __name__ == "__main__":
    name = "Profit-v0"
    register_env(name)
    env = gym.make(
        name,
        width=MAX_WIDTH,
        height=MAX_HEIGHT,
        turns=50,
        products={},
        render_mode=None,
    )
    # check_env(env)
    run_gym(env)
