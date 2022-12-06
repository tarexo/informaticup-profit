from model import ActorCritic, DeepQNetwork
from profit_gym import register_gym, make_gym
from helper.constants.settings import *

import numpy as np
import tensorflow as tf

import tqdm
import collections
import statistics


def test_model_sanity(env, model):
    state, _ = env.reset(obstacle_probability=model.obstacle_probability)

    for _ in range(MAX_STEPS_EACH_EPISODE):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        greedy_action = model.verbose_greedy_prediction(state)
        state, reward, done, legal, info = env.step(greedy_action)
        print("reward:", reward)

        if not legal:
            best_building = env.get_building_from_action(greedy_action)
            print("\nillegal building predicted: ")
            print(best_building)
        if done or not legal:
            break

    env.render()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(42)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)
    eps = np.finfo(np.float32).eps.item()

    register_gym("Profit-v0")
    env = make_gym("Profit-v0")

    model = DeepQNetwork(env)
    # model = ActorCritic(env)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    min_episodes = min(500, int(0.1 * MAX_EPISODES))
    model_sanity_check_frequency = 200
    solved_reward_threshold = 0.95 * SUCCESS_REWARD

    running_rewards = collections.deque(maxlen=min_episodes)
    progress = tqdm.trange(MAX_EPISODES)

    for episode in progress:
        if episode % model_sanity_check_frequency == 0:
            test_model_sanity(env, model)

        with tf.GradientTape() as tape:
            loss, episode_reward = model.run_episode(episode)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_rewards.append(episode_reward)
        running_mean_reward = statistics.mean(running_rewards)

        progress_info = collections.OrderedDict()
        progress_info["p"] = "%.2f" % model.obstacle_probability
        progress_info["ε"] = "%.2f" % model.exploration_rate
        progress_info["mean_final_reward"] = "%.2f" % running_mean_reward
        progress.set_postfix(progress_info)

        if running_mean_reward > solved_reward_threshold and episode >= min_episodes:
            break
