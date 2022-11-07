import gym
from gym.envs.registration import register
from gym.utils.env_checker import check_env


def register_env(name):
    register(
        id="Profit-v0",
        entry_point="environment:Environment",
        max_episode_steps=300,
    )


def run_gym(env, seed=None):
    env.action_space.seed(seed)

    observation, info = env.reset(seed=seed)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    env_name = "Profit-v0"
    register_env(env_name)
    env = gym.make(
        env_name, width=100, height=100, turns=50, products={}, render_mode="human"
    )
    # check_env(env)
    run_gym(env)
