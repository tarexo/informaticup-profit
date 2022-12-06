import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Input,
)
import numpy as np

from helper.constants.settings import *


class BaseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.model = None

    def create(self):
        raise NotImplementedError

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save(path)

    def summary(self):
        if self.model is None:
            self.create()
        self.model.summary()

    @staticmethod
    def get_expected_return(rewards, normalize=False):
        returns = []
        discounted_sum = 0
        for reward in tf.cast(rewards[::-1], dtype=tf.float32):
            discounted_sum = reward + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)

        if normalize:
            returns = np.array(returns)
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

        return returns

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def run_episode(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        pass

    def verbose_greedy_prediction(self):
        pass

    def call(self, inputs):
        if self.model is None:
            self.create()

        return self.model(inputs)


class ActorCritic(BaseModel):
    def __init__(self):
        super().__init__()
        self.critic_loss_function = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def create(self):
        board_size = (MAX_HEIGHT, MAX_WIDTH, NUM_CHANNELS)
        action_size = NUM_ACTIONS

        # shared network
        inputs = Input(shape=board_size)
        x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), activation="relu")(inputs)

        # policy head
        p = Conv2D(filters=2, kernel_size=1, activation="relu")(x)
        p = Flatten()(p)
        p = policy_head = Dense(action_size, activation="softmax", name="policy")(p)

        # value head
        v = Conv2D(filters=1, kernel_size=1, activation="relu")(x)
        v = Flatten()(v)
        v = value_head = Dense(1, activation="tanh", name="value")(v)

        self.model = Model(inputs=inputs, outputs=[policy_head, value_head])

    def verbose_greedy_prediction(self, state):
        action_probs, value = self.model(state)
        greedy_action = np.argmax(action_probs)

        print("\naction_probs:")
        print(action_probs.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))
        print("value:", value.numpy()[0, 0])

        return greedy_action

    def compute_loss(self, action_probs, values, rewards):
        returns = self.get_expected_return(rewards)
        diff = np.array(returns) - np.array(values)

        actor_loss = tf.math.reduce_sum(-tf.math.log(action_probs) * diff)
        critic_loss = self.critic_loss_function(values, returns)

        return actor_loss + critic_loss

    def run_episode(self, env, model, max_steps):
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

        loss = self.compute_loss(episode_action_probs, episode_values, episode_rewards)
        return loss, episode_rewards[-1]


class DeepQNetwork(BaseModel):
    def __init__(self):
        super().__init__()
        self.loss_function = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.exploration_rate = 0.2
        tf.keras.optimizers.schedules.ExponentialDecay(
            self.exploration_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )

    def create(self):
        board_size = (MAX_HEIGHT, MAX_WIDTH, NUM_CHANNELS)
        action_size = NUM_ACTIONS

        inputs = Input(shape=board_size)
        x = Conv2D(filters=64, kernel_size=5, strides=(1, 1), activation="relu")(inputs)
        x = Dense(filters=64, kernel_size=1, activation="relu")(x)
        x = Flatten()(x)
        x = q_values = Dense(action_size, activation="linear", name="q-values")(x)

        self.model = Model(inputs=inputs, outputs=q_values)

    def verbose_greedy_prediction(self, state):
        q_values = self.model(state)
        greedy_action = np.argmax(q_values)

        print("\nq_values:")
        print(q_values.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS)))

        return greedy_action

    def compute_q_loss(self, q_values, rewards):
        returns = self.get_expected_return(rewards)
        loss = self.loss_function(q_values, returns)

        return loss

    def run_q_episode(self, env, model, max_steps):
        state, _ = env.reset()

        episode_q_values = []
        episode_rewards = []
        for step in range(max_steps):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            q_values = model(state)

            if np.random.rand() <= self.exploration_rate:
                action = np.random.choice(NUM_ACTIONS)
            else:
                action = np.argmax(q_values)

            state, reward, done, legal, info = env.step(action)

            episode_q_values.append(q_values[0, action])
            episode_rewards.append(reward)

            if done or not legal:
                break

        loss = self.compute_loss(episode_q_values, episode_rewards)
        return loss, episode_rewards[-1]
