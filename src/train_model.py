from settings import *
from model.architecture import ActorCritic, DeepQNetwork
from evaluate_models import *
from helper.profiling import profile
from environment.setup import set_default_options, make_gym

import numpy as np
import tensorflow as tf

import tqdm
import collections
import statistics
import os

from matplotlib import pyplot as plt


def setup_matplotlib():
    plt.ion()
    ax = plt.gca()
    ax.set_ylim([0.0, 1.0])
    plt.show()


def plot(train_history, val_history):
    plt.plot(*zip(*train_history), "b")
    plt.plot(*zip(*val_history), "g")
    plt.draw()
    plt.pause(0.001)


def determine_difficulty(mean_reward):
    if mean_reward <= INCREASE_DIFFICULTY_AT:
        return 0.0
    elif mean_reward >= MAX_DIFFICULTY_AT:
        return 1.0
    return (mean_reward - INCREASE_DIFFICULTY_AT) / (
        MAX_DIFFICULTY_AT - INCREASE_DIFFICULTY_AT
    )


@profile
def train(env, model, max_episodes):
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    mean_val_reward = 0.0
    mean_train_reward = 0.0
    val_rewards = collections.deque(maxlen=min_episodes // model_val_frequency)
    train_rewards = collections.deque(maxlen=min_episodes)

    train_history = []
    val_history = []

    progress = tqdm.trange(max_episodes)
    for episode in progress:
        difficulty = determine_difficulty(mean_val_reward)
        if model.architecture_name == "A-C":
            exploration_rate = 0.05
        else:
            exploration_rate = FINAL_EXPLORATION_RATE ** (episode / max_episodes)

        if episode % model_val_frequency == 0:
            val_reward = evaluate(env, model, difficulty, num_episodes=1)
            val_rewards.append(val_reward)
            mean_val_reward = statistics.mean(val_rewards)

            train_history.append([episode, mean_train_reward])
            val_history.append([episode, mean_val_reward])

            plot(train_history, val_history)

        if episode % model_sanity_check_frequency == 0:
            check_model_sanity(env, model, difficulty)

        with tf.GradientTape() as tape:
            state, _ = env.reset(difficulty=difficulty)

            loss, episode_reward = model.run_episode(state, exploration_rate)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_rewards.append(episode_reward)
        mean_train_reward = statistics.mean(train_rewards)

        progress_info = collections.OrderedDict()
        progress_info["diff"] = "%.2f" % difficulty
        progress_info["ε"] = "%.2f" % exploration_rate
        progress_info["train_reward"] = "%.2f" % mean_train_reward
        progress_info["val_reward"] = "%.2f" % mean_val_reward
        progress.set_postfix(progress_info)

        if (
            mean_val_reward > solved_reward_threshold
            and difficulty == 1.0
            and episode > min_episodes
        ):
            break


def train_model(width, height, field_of_vision, transfer_model_path=None):
    env = make_gym(width, height, field_of_vision)

    if MODEL_ID == "DQN":
        model = DeepQNetwork(env)
    elif MODEL_ID == "A-C":
        model = ActorCritic(env)
    model.create()

    model_path = model.get_model_path()
    if os.path.isdir(model_path):
        print(f"{model.get_model_description()} has already been trained!")
        print("remove it from the 'saved_model'-directory in order to retrain")
        return model_path

    print(f"\nTraining {model.get_model_description()}...\n")
    if transfer_model_path is not None:
        model.transfer(transfer_model_path, trainable=True)
    model.summary()

    # Main Training
    train(env, model, MAX_EPISODES)
    model.save(model_path)
    evaluate_model(env, model)

    # Fine Tune
    if model.has_frozen_layers():
        model.unfreeze()
        train(env, model, FINE_TUNE_EPISODES)
        model.save(model_path)
        evaluate_model(env, model)

    return model_path


def train_transfer_models(width, height):
    transfer_model_path = None

    field_of_vision_sizes = [5, 7, 9]

    print(f"\nTraining multiple transfer models...\n")
    for field_of_vision in field_of_vision_sizes:
        model_path = train_model(width, height, field_of_vision, transfer_model_path)
        transfer_model_path = model_path


if __name__ == "__main__":
    set_default_options()

    min_episodes = max(500, int(0.2 * MAX_EPISODES))
    solved_reward_threshold = 0.98 * SUCCESS_REWARD
    model_val_frequency = 10
    model_sanity_check_frequency = 100
    model_save_frequency = 2500

    width = height = 30
    field_of_vision = 15
    transfer_model_path = None
    train_model(width, height, field_of_vision, transfer_model_path)
