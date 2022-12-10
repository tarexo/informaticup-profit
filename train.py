from model import ActorCritic, DeepQNetwork
from profit_gym import register_gym, make_gym
from helper.constants.settings import *
from helper.dicts.convert_actions import action_to_description

import numpy as np
import tensorflow as tf

import tqdm
import collections
import statistics
import os


def test_model_sanity(env, model, difficulty, no_obstacles):
    state, _ = env.reset(difficulty=difficulty, no_obstacles=no_obstacles)

    for _ in range(MAX_STEPS_EACH_EPISODE):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        greedy_action = model.verbose_greedy_prediction(state)
        state, reward, done, legal, info = env.step(greedy_action)

        direction_id, subbuilding_id = env.split_action(greedy_action)
        action_description = action_to_description(direction_id, subbuilding_id)
        print(
            f"\nGreedy Action: {action_description}"
            + (" (illegal)" if not legal else "")
        )
        print("--> Reward:", reward)
        if done or not legal:
            break

    env.render()


def determine_difficulty(mean_reward):
    if mean_reward <= INCREASE_DIFFICULTY_AT:
        return 0.0
    elif mean_reward >= MAX_DIFFICULTY_AT:
        return 1.0
    return (mean_reward - INCREASE_DIFFICULTY_AT) / (
        MAX_DIFFICULTY_AT - INCREASE_DIFFICULTY_AT
    )


def train(env, model, max_episodes, no_obstacles=False):
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    running_rewards = collections.deque(maxlen=min_episodes)
    running_mean_reward = 0.0
    progress = tqdm.trange(max_episodes)
    for episode in progress:
        difficulty = determine_difficulty(running_mean_reward)
        exploration_rate = FINAL_EXPLORATION_RATE ** (episode / max_episodes)

        if episode % model_sanity_check_frequency == 0:
            test_model_sanity(env, model, difficulty, no_obstacles)

        with tf.GradientTape() as tape:
            state, _ = env.reset(difficulty=difficulty, no_obstacles=no_obstacles)

            loss, episode_reward = model.run_episode(state, exploration_rate)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_rewards.append(episode_reward)
        running_mean_reward = statistics.mean(running_rewards)

        progress_info = collections.OrderedDict()
        progress_info["difficulty"] = "%.2f" % difficulty
        progress_info["ε"] = "%.2f" % exploration_rate
        progress_info["mean_reward"] = "%.2f" % running_mean_reward
        progress.set_postfix(progress_info)

        if running_mean_reward > solved_reward_threshold and episode >= min_episodes:
            break


def train_model(width, height, num_conv_layers, transfer_model_path=None):
    env = make_gym(width, height)

    if MODEL_ID == "DQN":
        model = DeepQNetwork(env)
    elif MODEL_ID == "A-C":
        model = ActorCritic(env)
    model.create(num_conv_layers)

    model_path = model.get_model_path()
    if os.path.isdir(model_path):
        print(f"{model.get_model_description()} has already been trained!")
        return model_path

    print(f"\nTraining {model.get_model_description()}...\n")
    if transfer_model_path is not None:
        model.transfer(transfer_model_path, trainable=False)
    model.summary()

    # Pre-Train (no obstacles)
    # train(env, model, PRE_TRAIN_EPISODES, no_obstacles=True)
    # model.save(model_path)

    # Main Training
    train(env, model, MAX_EPISODES)
    model.save(model_path)

    # Fine Tune
    if model.has_frozen_layers():
        model.unfreeze()
        train(env, model, FINE_TUNE_EPISODES)
        model.save(model_path)

    return model_path


def train_transfer_models(initial_size, final_size):
    transfer_model_path = None

    step_size = KERNEL_SIZE - 1

    print(f"\nTraining multiple transfer models...\n")
    for size in range(initial_size, final_size + 1, step_size):
        num_conv_layers = (size + 1) // step_size
        model_path = train_model(size, size, num_conv_layers, transfer_model_path)
        transfer_model_path = model_path


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    np.random.seed(42)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)
    # suppress AVX_WARNING!
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    min_episodes = min(500, int(0.1 * MAX_EPISODES))
    solved_reward_threshold = 0.98 * SUCCESS_REWARD
    model_sanity_check_frequency = 200
    model_save_frequency = 2500

    register_gym()

    if TRANSFER_LEARNING:
        initial_size = 2 + KERNEL_SIZE
        final_size = 13
        train_transfer_models(initial_size, final_size)
    else:
        width = height = 3
        num_conv_layers = (width + 1) // (KERNEL_SIZE - 1)
        transfer_model_path = None
        # transfer_model_path = ".\\saved_models\\SIMPLE__5x5__DQN_128-3x3_64"

        train_model(width, height, num_conv_layers, transfer_model_path)
