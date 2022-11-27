from profit_gym import register_gym, make_gym
from helper.constants.settings import *

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import tqdm
import collections
import statistics


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = x = layers.Input(shape=((MAX_HEIGHT, MAX_WIDTH, NUM_CHANNELS)))

    # Convolutions on the frames on the screen
    x = layers.Conv2D(128, 2, strides=1, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 2, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 2, strides=2, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    # x = layers.Dense(128, activation="relu")(x)
    actions = x = layers.Dense(NUM_ACTIONS, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=actions)


def test_model_sanity(env, q_model, num_steps=1):
    state, _ = env.reset()
    env.render()

    for _ in range(num_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        q_values = q_model(state)
        greedy_action = np.argmax(q_values)

        best_building = env.get_building_from_action(greedy_action)

        print("q_values:")
        print(q_values.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))

        state, done, reward, legal, info = env.step(greedy_action)

        if legal:
            env.render()
        else:
            print("illegal building predicted: ")
            print(best_building)
            break


if __name__ == "__main__":

    register_gym("Profit-v0")
    env = make_gym("Profit-v0")
    q_model = create_q_model()
    q_model.summary()

    np.set_printoptions(precision=2, suppress=True)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Initialize parameters
    max_episodes = 10000
    min_episodes = 500

    model_sanity_check_frequency = 100
    solved_reward_threshold = 0.9 * SUCCESS_REWARD
    exploration_rate = 0.6

    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    running_rewards = collections.deque(maxlen=min_episodes)
    progress = tqdm.trange(max_episodes)

    for i in progress:
        state, _ = env.reset()
        with tf.GradientTape() as tape:
            if i % model_sanity_check_frequency == 0:
                test_model_sanity(env, q_model)

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Obtain Q-values from model
            q_values = q_model(state)

            epsilon = np.random.rand()
            if epsilon <= exploration_rate:
                # Select random action
                action = np.random.choice(NUM_ACTIONS)
            else:
                # Select action with highest q-value
                action = np.argmax(q_values)

            state, reward, done, legal, info = env.step(action)

            # Obtain Q-value
            q_value = q_values[0, action]

            # Compute loss value
            loss = loss_function([[reward]], [[q_value]])

            # Compute gradients
            grads = tape.gradient(loss, q_model.trainable_variables)

            # Apply gradients to update model weights
            opt.apply_gradients(zip(grads, q_model.trainable_variables))

            running_rewards.append(reward)
            running_mean_reward = statistics.mean(running_rewards)
            progress.set_postfix(
                running_reward="%.3f" % running_mean_reward,
                exploration_rate="%.3f" % exploration_rate,
            )

            if (i + 1) % int(0.1 * max_episodes) == 0:
                exploration_rate /= 1.5

            if running_mean_reward > solved_reward_threshold and i >= min_episodes:
                break
