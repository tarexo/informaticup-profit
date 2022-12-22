import tensorflow as tf
from keras.models import Model, load_model, save_model
from keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Input,
)
import os
import numpy as np

from helper.constants.settings import *
from helper.dicts.convert_actions import named_array
from mcts import MonteCarloTreeSearch, copy_env
from copy import deepcopy
from helper.constants.settings import NUM_ACTIONS
from random import shuffle
from tqdm import tqdm
import random


class BaseModel(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.model = None
        self.env = env
        self.board_size = self.env.observation_shape
        self.action_size = NUM_ACTIONS

        self.obstacle_probability = 0.0
        self.exploration_rate = None

    def create(self, num_conv_layers):
        x = inputs = Input(shape=self.board_size, name="Observation")

        num_filters = INITIAL_CONV_FILTERS
        for i in range(num_conv_layers):
            x = Conv2D(num_filters, 3, name=f"Conv_{i+1}", activation="relu")(x)
            if FILTER_DECREASING:
                num_filters //= 2

        x = Flatten(name="Flatten")(x)
        x = Dense(units=NUM_FEATURES, activation="relu", name="Features")(x)

        outputs = self.create_heads(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.architecture_name)

    def transfer(self, path, trainable):
        if not os.path.isdir(path):
            print(f"WARNING: model could not be tranfered. {path} does not exist!")
            return
        trained_model = load_model(path, compile=False)
        conv_layers = [layer for layer in trained_model.layers if type(layer) == Conv2D]

        for i, trained_layer in enumerate(conv_layers, 1):
            if self.model.layers[i].filters != trained_layer.filters:
                print(f"WARNING: #filters of Convolutional Layer {i} does not match")
                print(f"{self.model.layers[i].filters} vs {trained_layer.filters}")
                continue
            self.model.layers[i].set_weights(trained_layer.get_weights())
            self.model.layers[i].trainable = trainable
            print(f"Convolutional Layer {i} has been transfered from {path}")

        if RETRAIN_LAST_CONV_LAYER:
            self.model.layers[i].trainable = True

    def load(self, path):
        if not os.path.isdir(path):
            print(f"WARNING: model could not be loaded. {path} does not exist!")
            return
        self.get_model_path = load_model(path, compile=False)
        print(f"Model has been loaded from {path}")

    def save(self, path):
        save_model(self.model, path, include_optimizer=False)
        print(f"model has been saved to {path}")

    def get_model_description(self):
        conv_layers = [layer for layer in self.model.layers if type(layer) == Conv2D]
        conv_str = ""
        for conv_layer in conv_layers:
            num_filters = conv_layer.filters
            conv_str += f"_{num_filters}"

        return f"{self.env.width}x{self.env.height}__{self.architecture_name}{conv_str}"

    def get_model_path(self):
        return os.path.join(".", "saved_models", self.get_model_description())

    def summary(self):
        print()
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
        return self.model(inputs)


class ActorCritic(BaseModel):
    def __init__(self, env):
        super().__init__(env)
        self.architecture_name = "A-C"
        self.critic_loss_function = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def create_heads(self, x):
        # unique policy network
        p = Dense(units=NUM_FEATURES * 2, activation="relu", name="Policy-Features")(x)
        p = Dense(self.action_size, activation="softmax", name="Policy-Head")(p)

        # unique value network
        v = Dense(units=NUM_FEATURES, activation="relu", name="Value-Features")(x)
        v = Dense(1, activation="tanh", name="Value-Head")(v)

        return [p, v]

    def verbose_greedy_prediction(self, state):
        action_probs, value = self.model(state)
        greedy_action = np.argmax(action_probs)

        named_action_probs = named_array(action_probs)
        print("\n\nAction Policy:")
        print(named_action_probs)
        print("\nValue Prediction:", round(value.numpy()[0, 0], 3))

        return greedy_action

    def compute_loss(self, action_probs, values, rewards):
        returns = self.get_expected_return(rewards)
        advantage = np.array(returns) - np.array(values)

        actor_loss = tf.math.reduce_sum(-tf.math.log(action_probs) * advantage)
        critic_loss = self.critic_loss_function(values, returns)

        return actor_loss + critic_loss

    def run_episode(self, episode, obstacle_probability):
        # self.obstacle_probability = FINAL_OBSTACLE_PROBABILITY * episode / MAX_EPISODES
        self.obstacle_probability = min(MAX_OBSTACLE_PROBABILITY, obstacle_probability)

        state, _ = self.env.reset(obstacle_probability=self.obstacle_probability)

        episode_action_probs = []
        episode_values = []
        episode_rewards = []
        for step in range(MAX_STEPS_EACH_EPISODE):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, value = self.model(state)
            action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))

            state, reward, done, legal, info = self.env.step(action)

            episode_action_probs.append(action_probs[0, action])
            episode_values.append(value[0, 0])
            episode_rewards.append(reward)

            if done or not legal:
                break

        loss = self.compute_loss(episode_action_probs, episode_values, episode_rewards)
        return loss, episode_rewards[-1]

    def gather_examples(self, num_episodes):
        train_examples = []

        for _ in tqdm(range(num_episodes)):
            episode_example = self.collect_experience()
            train_examples.extend(episode_example)

        shuffle(train_examples)
        return train_examples

    def collect_experience(self):
        # Credit to Josh Varty: https://github.com/JoshVarty/AlphaZeroSimple/blob/master/trainer.py
        train_examples = []

        state, _ = self.env.reset(obstacle_probability=self.obstacle_probability)

        # loop:
        while True:
            # - save game state
            board_state = state
            # - run mcts for set num of episodes
            mcts = MonteCarloTreeSearch(deepcopy(self.env), state, self.model)
            root, action = mcts.run(num_runs=50)
            # - transform N of root to action probabilities
            action_probs = [0 for _ in range(NUM_ACTIONS)]
            for child in root.children:
                i = child.action_taken
                v = child.N
                action_probs[i] = v
            action_probs = action_probs / np.sum(action_probs)
            # - Store: (game state, action probs)
            train_examples.append((board_state, action_probs))
            # TODO At some point no valid child is found eventhough there are legal actions left
            # ? How do we handle dead ends in the tree?
            # - Take an action based on the root node
            state, reward, done, legal, info = self.env.step(action)
            self.env.render()

            # - If reward == SUCCESS_REWARD [CUSTOM]: or no more moves possible
            if reward == SUCCESS_REWARD:
                ret = []
                for board_state, action_probs in train_examples:
                    ret.append((board_state, action_probs, reward))
                return ret

            if not self.legal_action_possible(copy_env(self.env)):
                ret = []
                for board_state, action_probs in train_examples:
                    ret.append((board_state, action_probs, -1))
                return ret

    def legal_action_possible(self, env):
        for action in range(NUM_ACTIONS):
            _env = copy_env(env)
            state, reward, done, legal, info = _env.step(action)
            if legal:
                return True
        return False


class DeepQNetwork(BaseModel):
    def __init__(self, env):
        super().__init__(env)
        self.architecture_name = "DQN"
        self.loss_function = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def create_heads(self, x):
        # single q-values head
        q_values = Dense(self.action_size, activation="linear", name="Q-Values")(x)
        return q_values

    def verbose_greedy_prediction(self, state):
        q_values = self.model(state)
        greedy_action = np.argmax(q_values)

        named_q_values = named_array(q_values)
        print("\n\nQ-Values:")
        print(named_q_values)

        return greedy_action

    def compute_loss(self, q_values, rewards):
        returns = self.get_expected_return(rewards)
        loss = self.loss_function(q_values, returns)

        return loss

    def run_episode(self, episode, obstacle_probability):
        # self.obstacle_probability = FINAL_OBSTACLE_PROBABILITY * episode / MAX_EPISODES
        self.obstacle_probability = min(MAX_OBSTACLE_PROBABILITY, obstacle_probability)
        self.exploration_rate = 0.5 ** (episode / (0.1 * MAX_EPISODES))

        state, _ = self.env.reset(obstacle_probability=self.obstacle_probability)

        episode_q_values = []
        episode_rewards = []
        for step in range(MAX_STEPS_EACH_EPISODE):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            q_values = self.model(state)

            if np.random.rand() <= self.exploration_rate:
                action = np.random.choice(NUM_ACTIONS)
            else:
                action = np.argmax(q_values)

            state, reward, done, legal, info = self.env.step(action)

            episode_q_values.append(q_values[0, action])
            episode_rewards.append(reward)

            if done or not legal:
                break

        loss = self.compute_loss(episode_q_values, episode_rewards)
        return loss, episode_rewards[-1]
