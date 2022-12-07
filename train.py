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


def test_model_sanity(env, model):
    state, _ = env.reset(obstacle_probability=model.obstacle_probability)

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


def train(env, model):
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    running_rewards = collections.deque(maxlen=min_episodes)
    running_mean_reward = 0.0
    progress = tqdm.trange(MAX_EPISODES)
    for episode in progress:
        if episode % model_sanity_check_frequency == 0:
            test_model_sanity(env, model)

        with tf.GradientTape() as tape:
            loss, episode_reward = model.run_episode(episode, 0.5 * running_mean_reward)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_rewards.append(episode_reward)
        running_mean_reward = statistics.mean(running_rewards)

        progress_info = collections.OrderedDict()
        # currently uses mean_final_reward  as obstacle_probability
        # progress_info["p"] = "%.2f" % model.obstacle_probability
        if model.exploration_rate is not None:
            progress_info["ε"] = "%.2f" % model.exploration_rate
        progress_info["mean_final_reward"] = "%.2f" % running_mean_reward
        progress.set_postfix(progress_info)

        if running_mean_reward > solved_reward_threshold and episode >= min_episodes:
            break


def train_model(width, height, num_conv_layers, transfer_model_path=None):
    env = make_gym(width, height)

    if MODEL_ID == "QDN":
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

    train(env, model)
    model.save(model_path)

    return model_path


def train_transfer_models(board_sizes):
    transfer_model_path = None

    print(f"\nTraining multiple transfer models...\n")
    for num_conv_layers, size in enumerate(board_sizes, 2):
        model_path = train_model(size, size, num_conv_layers, transfer_model_path)
        transfer_model_path = model_path


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    np.random.seed(42)
    tf.random.set_seed(42)
    eps = np.finfo(np.float32).eps.item()
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)
    # supress AVX_WARNING!
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    min_episodes = min(500, int(0.1 * MAX_EPISODES))
    solved_reward_threshold = 0.98 * SUCCESS_REWARD
    model_sanity_check_frequency = 200
    model_save_frequency = 2500

    register_gym()

    if TRANSFER_LEARNING:
        board_sizes = range(4, 12, 2)
        train_transfer_models(board_sizes)
    else:
        width = height = 16
        num_conv_layers = 5
        # transfer_model_path = None
        # transfer_model_path = ".\\saved_models\\4x4__DQN_512_256"
        # transfer_model_path = ".\\saved_models\\6x6__DQN_512_256_128"
        transfer_model_path = ".\\saved_models\\8x8__DQN_512_256_128_64"

        train_model(width, height, num_conv_layers, transfer_model_path)
