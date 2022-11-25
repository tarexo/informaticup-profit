import numpy as np
import tensorflow as tf
import collections
import tqdm
import statistics

from model import *
from profit_gym import register_gym, make_gym
from helper.dicts.convert_actions import *
from helper.constants.settings import *

# used in compute_loss()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def run_episode(env, model, max_steps):
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    state, _ = env.reset()

    for t in tf.range(max_steps):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action_logits_t, value_prediction = model(state)

        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        state, reward, done, legal, info = env.step(action)

        values = values.write(t, tf.squeeze(value_prediction))
        action_probs = action_probs.write(t, action_probs_t[0, action])
        rewards = rewards.write(t, reward)

        if done or legal:
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(rewards, gamma, standardize=True):
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (
            tf.math.reduce_std(returns) + np.finfo(np.float32).eps.item()
        )

    return returns


def compute_loss(action_probs, values, returns):
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


@tf.function
def train_step(env, model, optimizer, gamma, max_steps_per_episode):
    """Runs a model training step."""

    with tf.GradientTape() as tape:
        action_probs, values, rewards = run_episode(env, model, max_steps_per_episode)
        returns = get_expected_return(rewards, gamma)
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]
        ]
        loss = compute_loss(action_probs, values, returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return int(episode_reward)


def test_model_sanity(env, model):
    state, _ = env.reset()
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)
    action_logits_t, value_pred = model(state)
    action_distribution = tf.nn.softmax(action_logits_t)
    greedy_action = np.argmax(action_distribution.numpy())

    best_building = env.get_building_from_action(greedy_action)

    print("action distribution:", action_distribution.numpy())
    print("value prediction:", value_pred.numpy())

    env.render()
    if env.is_legal_position(best_building):
        env.add_building(best_building)
        env.render()
    else:
        print("illegal building predicted: ")
        print(best_building)


def train():
    register_gym("Profit-v0")
    env = make_gym("Profit-v0")
    model = ActorCritic(CONV_SIZE, CONV_DEPTH, DENSE_SIZE, DENSE_DEPTH)
    # model.summary()

    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)

    tf.random.set_seed(42)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    min_episodes_criterion = 100
    max_episodes = 1000
    max_steps_per_episode = 1

    reward_threshold = 0.99
    running_reward = 0

    gamma = 0.99
    episodes_reward: collections.deque = collections.deque(
        maxlen=min_episodes_criterion
    )

    t = tqdm.trange(max_episodes)
    for i in t:
        episode_reward = train_step(env, model, optimizer, gamma, max_steps_per_episode)

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

    print(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!")

    test_model_sanity(env, model)
    # test_model_sanity(env, model)
    # test_model_sanity(env, model)


if __name__ == "__main__":
    train()
