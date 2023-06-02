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
from datetime import datetime

logdir = "logs"# + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

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
    mean_test_reward = 0.0
    mean_train_reward = 0.0
    test_rewards = collections.deque(maxlen=min_episodes // model_test_frequency)
    train_rewards = collections.deque(maxlen=min_episodes)

    progress = tqdm.trange(max_episodes)
    for episode in progress:
        difficulty = determine_difficulty(mean_test_reward)
        if model.architecture_name == "A-C":
            exploration_rate = 0.05
        else:
            exploration_rate = FINAL_EXPLORATION_RATE ** (episode / max_episodes)

        if episode % model_test_frequency == 0:
            test_reward = evaluate(env, model, difficulty, num_episodes=1)
            test_rewards.append(test_reward)
            mean_test_reward = statistics.mean(test_rewards)

        if episode % model_sanity_check_frequency == 0:
            check_model_sanity(env, model, difficulty)

        with tf.GradientTape() as tape:
            state, _ = env.reset(difficulty=difficulty)

            loss, episode_reward = model.run_episode(state, exploration_rate)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_rewards.append(episode_reward)
        mean_train_reward = statistics.mean(train_rewards)

        with file_writer.as_default():
            tf.summary.scalar("train/train_reward", mean_train_reward, step=episode)
            tf.summary.scalar("val/val_reward", mean_test_reward, step=episode)
            tf.summary.scalar("train/epsilon", exploration_rate, step=episode)

        progress_info = collections.OrderedDict()
        progress_info["diff"] = "%.2f" % difficulty
        progress_info["ε"] = "%.2f" % exploration_rate
        progress_info["train_reward"] = "%.2f" % mean_train_reward
        progress_info["test_reward"] = "%.2f" % mean_test_reward

        progress.set_postfix(progress_info)

        if (
            mean_test_reward > solved_reward_threshold
            and difficulty == 1.0
            and episode > min_episodes
        ):
            break


def train_model(width, height, field_of_vision, transfer_model_path=None, logdir ="logs"):
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
    logdir = "logs"
    min_episodes = max(500, int(0.2 * MAX_EPISODES))
    solved_reward_threshold = 0.98 * SUCCESS_REWARD
    model_test_frequency = 10
    model_sanity_check_frequency = 100
    model_save_frequency = 2500

    width = height = 30
    field_of_vision = 15
    transfer_model_path = None
    train_model(width, height, field_of_vision, transfer_model_path,logdir)
