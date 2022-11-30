from alphazero import AlphaZero
from profit_gym import register_gym, make_gym
from helper.constants.settings import *

import numpy as np
import tensorflow as tf

import tqdm
import collections
import statistics


def test_model_sanity(env, model, num_steps):
    state, _ = env.reset()

    for _ in range(num_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs, value = model(state)
        greedy_action = np.argmax(action_probs)
        state, reward, done, legal, info = env.step(greedy_action)

        print("\nvalue:", value.numpy()[0, 0])
        print("reward:", reward)
        print("action_probs:")
        print(action_probs.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))

        if not legal:
            best_building = env.get_building_from_action(greedy_action)
            print("\nillegal building predicted: ")
            print(best_building)
        if done or not legal:
            break

    env.render()


def get_expected_return(rewards, normalize=False):
    """Compute expected returns per timestep."""

    returns = []
    discounted_sum = 0
    for reward in tf.cast(rewards[::-1], dtype=tf.float32):
        discounted_sum = reward + GAMMA * discounted_sum
        returns.insert(0, discounted_sum)

    if normalize:
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

    return returns


def compute_loss(action_probs, values, rewards):
    """Computes the combined Actor-Critic loss."""

    returns = get_expected_return(rewards)
    diff = np.array(returns) - np.array(values)

    actor_loss = tf.math.reduce_sum(-tf.math.log(action_probs) * diff)
    critic_loss = critic_loss_function(values, returns)

    return actor_loss + critic_loss


def run_episode(env, model, max_steps):
    state, _ = env.reset()

    episode_action_probs = []
    episode_values = []
    episode_rewards = []
    for step in range(max_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs, value = model(state)

        action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))

        state, reward, done, legal, info = env.step(action)

        episode_action_probs.append(action_probs[0, action])
        episode_values.append(value[0, 0])
        episode_rewards.append(reward)

        if done or not legal:
            break

    loss = compute_loss(episode_action_probs, episode_values, episode_rewards)

    return loss, episode_rewards[-1]


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(42)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)
    eps = np.finfo(np.float32).eps.item()

    register_gym("Profit-v0")
    env = make_gym("Profit-v0")

    model = AlphaZero(blocks=4, block_filters=128).model
    model.summary()

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    critic_loss_function = tf.keras.losses.Huber(
        reduction=tf.keras.losses.Reduction.SUM
    )

    max_episodes = 2000
    min_episodes = 200

    max_steps_each_episode = 3

    model_sanity_check_frequency = 50
    solved_reward_threshold = 0.95 * SUCCESS_REWARD

    running_rewards = collections.deque(maxlen=min_episodes)
    progress = tqdm.trange(max_episodes)

    for episode in progress:
        if episode % model_sanity_check_frequency == 0:
            test_model_sanity(env, model, max_steps_each_episode)

        with tf.GradientTape() as tape:
            loss, episode_reward = run_episode(env, model, max_steps_each_episode)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_rewards.append(episode_reward)
        running_mean_reward = statistics.mean(running_rewards)
        progress.set_postfix(running_reward="%.2f" % running_mean_reward)

        if running_mean_reward > solved_reward_threshold and episode >= min_episodes:
            break
