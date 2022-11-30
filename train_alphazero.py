from alphazero import AlphaZero, DQN
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

        print("\naction_probs:")
        print(action_probs.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))
        print("value:", value.numpy()[0, 0])
        print("reward:", reward)

        if not legal:
            best_building = env.get_building_from_action(greedy_action)
            print("\nillegal building predicted: ")
            print(best_building)
        if done or not legal:
            break

    env.render()


def test_q_model_sanity(env, model, num_steps):
    state, _ = env.reset()

    for _ in range(num_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        q_values = model(state)
        greedy_action = np.argmax(q_values)
        state, reward, done, legal, info = env.step(greedy_action)

        print("\nq_values:")
        print(q_values.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))
        print("reward:", reward)

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


def compute_q_loss(q_values, rewards):
    """Computes the combined Actor-Critic loss."""

    returns = get_expected_return(rewards)
    loss = critic_loss_function(q_values, returns)

    return loss


def run_episode(env, model, max_steps, exploration_rate):
    state, _ = env.reset()

    episode_action_probs = []
    episode_values = []
    episode_rewards = []
    for step in range(max_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_probs, value = model(state)

        if np.random.rand() <= 0.0:
            action = np.random.choice(NUM_ACTIONS)
        else:
            action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))

        state, reward, done, legal, info = env.step(action)

        episode_action_probs.append(action_probs[0, action])
        episode_values.append(value[0, 0])
        episode_rewards.append(reward)

        if done or not legal:
            break

    loss = compute_loss(episode_action_probs, episode_values, episode_rewards)
    return loss, episode_rewards[-1]


def run_q_episode(env, model, max_steps, exploration_rate):
    state, _ = env.reset()

    episode_q_values = []
    episode_rewards = []
    for step in range(max_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        q_values = model(state)

        if np.random.rand() <= exploration_rate:
            action = np.random.choice(NUM_ACTIONS)
        else:
            action = np.argmax(q_values)

        state, reward, done, legal, info = env.step(action)

        episode_q_values.append(q_values[0, action])
        episode_rewards.append(reward)

        if done or not legal:
            break

    loss = compute_q_loss(episode_q_values, episode_rewards)
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

    model = AlphaZero(blocks=8).model
    # model = DQN(blocks=8).model
    model.summary()

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    critic_loss_function = tf.keras.losses.Huber(
        reduction=tf.keras.losses.Reduction.SUM
    )

    max_episodes = 1000
    min_episodes = int(0.2 * max_episodes)

    max_steps_each_episode = 3
    exploration_rate = 0.6

    model_sanity_check_frequency = 50
    solved_reward_threshold = 0.95 * SUCCESS_REWARD

    running_rewards = collections.deque(maxlen=min_episodes)
    progress = tqdm.trange(max_episodes)

    for episode in progress:
        if episode % model_sanity_check_frequency == 0:
            test_model_sanity(env, model, max_steps_each_episode)

        with tf.GradientTape() as tape:
            loss, episode_reward = run_episode(
                env, model, max_steps_each_episode, exploration_rate
            )

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_rewards.append(episode_reward)
        running_mean_reward = statistics.mean(running_rewards)
        progress.set_postfix(
            running_reward="%.2f" % running_mean_reward,
            exploration_rate="%.2f" % exploration_rate,
        )

        if (episode + 1) % int(0.1 * max_episodes) == 0:
            exploration_rate /= 1.5

        if running_mean_reward > solved_reward_threshold and episode >= min_episodes:
            break
