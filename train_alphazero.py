from alphazero import AlphaZero
from profit_gym import register_gym, make_gym
from helper.constants.settings import *

import numpy as np
import tensorflow as tf

import tqdm
import collections
import statistics

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def test_model_sanity(env, model, num_steps=1):
    state, _ = env.reset()
    env.render()

    for _ in range(num_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probs, value = model(state)
        greedy_action = np.argmax(action_probs)

        best_building = env.get_building_from_action(greedy_action)

        print("value:", value.numpy()[0])
        print("action_probs:")
        print(action_probs.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))

        state, done, reward, legal, info = env.step(greedy_action)

        if legal:
            env.render()
        else:
            print("illegal building predicted: ")
            print(best_building)
            break


def compute_loss(action_prob, value, reward):
    """Computes the combined Actor-Critic loss."""

    diff = reward - value
    actor_loss = -tf.math.log(action_prob) * diff
    critic_loss = huber_loss([value], [reward])

    return actor_loss + critic_loss


if __name__ == "__main__":

    register_gym("Profit-v0")
    env = make_gym("Profit-v0")
    model = AlphaZero(blocks=2).model
    model.summary()

    np.set_printoptions(precision=2, suppress=True)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Initialize parameters
    max_episodes = 2000
    min_episodes = 500

    model_sanity_check_frequency = 100
    solved_reward_threshold = 0.9 * SUCCESS_REWARD
    exploration_rate = 0.0

    opt = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    running_rewards = collections.deque(maxlen=min_episodes)
    progress = tqdm.trange(max_episodes)

    for episode in progress:
        state, _ = env.reset()
        if episode % model_sanity_check_frequency == 0:
            test_model_sanity(env, model)

        with tf.GradientTape() as tape:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, value = model(state)

            epsilon = np.random.rand()
            if epsilon <= exploration_rate:
                # Select random action
                action = np.random.choice(NUM_ACTIONS)
            else:
                # Sample action from policy distribution
                action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))

            state, reward, done, legal, info = env.step(action)

            loss = compute_loss(action_probs[0, action], value[0, 0], reward)

            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

        running_rewards.append(reward)
        running_mean_reward = statistics.mean(running_rewards)
        progress.set_postfix(
            running_reward="%.3f" % running_mean_reward,
            exploration_rate="%.3f" % exploration_rate,
        )

        if (episode + 1) % int(0.1 * max_episodes) == 0:
            exploration_rate /= 2

        if running_mean_reward > solved_reward_threshold and i >= min_episodes:
            break
