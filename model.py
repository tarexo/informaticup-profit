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


class BaseModel(tf.keras.Model):
    def __init__(self, env):
        super().__init__()

        self.model = None
        self.env = env
        self.board_size = self.env.observation_shape
        self.action_size = NUM_ACTIONS

        self.eps = np.finfo(np.float32).eps.item()

    def create(self, num_conv_layers):
        x = field_of_vision = Input(shape=self.board_size, name="field_of_vision")
        target_position = Input(shape=(2,), name="target_position")

        for i in range(num_conv_layers):
            x = Conv2D(
                NUM_CONV_FILTERS,
                KERNEL_SIZE,
                name=f"Conv_{i+1}",
                activation="relu",
            )(x)

        obstacle_features = Flatten(name="Flatten")(x)

        x = tf.concat([obstacle_features, target_position], axis=1)
        x = Dense(units=NUM_FEATURES, activation="relu", name="Features")(x)

        outputs = self.create_heads(x)

        self.model = Model(
            inputs=[field_of_vision, target_position],
            outputs=outputs,
            name=self.architecture_name,
        )

    def transfer(self, path, trainable):
        if not os.path.isdir(path):
            print(f"WARNING: model could not be tranfered. {path} does not exist!")
            return
        trained_model = load_model(path, compile=False)
        conv_layers = [layer for layer in trained_model.layers if type(layer) == Conv2D]

        for i, trained_layer in enumerate(conv_layers, 1):
            if self.model.layers[i].filters != trained_layer.filters:
                print(f"WARNING: #filters of Convolutional Layer {i} do not match")
                print(f"{self.model.layers[i].filters} vs {trained_layer.filters}")
                continue
            self.model.layers[i].set_weights(trained_layer.get_weights())
            self.model.layers[i].trainable = trainable
            print(f"Convolutional Layer {i} has been frozen and transfered from {path}")

        self.model.layers[-2].set_weights(trained_model.layers[-2].get_weights())
        self.model.layers[-1].set_weights(trained_model.layers[-1].get_weights())

    def transfer_heads(self, trained_model):
        raise NotImplementedError

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
        env_str = "SIMPLE" if SIMPLE_GAME else "NORMAL"
        grid_str = f"{self.env.field_of_vision}x{self.env.field_of_vision}"
        architecture_str = self.architecture_name
        architecture_str += f"_{NUM_CONV_FILTERS}-{KERNEL_SIZE}x{KERNEL_SIZE}"
        architecture_str += f"_{NUM_FEATURES}"

        return env_str + "__" + grid_str + "__" + architecture_str

    def get_model_path(self):
        return os.path.join(".", "saved_models", self.get_model_description())

    def summary(self):
        print()
        self.model.summary()

    def has_frozen_layers(self):
        for layer in self.model.layers[1:]:
            if not layer.trainable:
                return True
        return False

    def unfreeze(self):
        for layer in self.model.layers[1:]:
            layer.trainable = True

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

    @staticmethod
    def min_max_scaling(reward):
        return (reward - ILLEGAL_REWARD) / (SUCCESS_REWARD - ILLEGAL_REWARD)

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def run_episode(self, *args, **kwargs):
        raise NotImplementedError

    def verbose_greedy_prediction(self, state):
        raise NotImplementedError

    def call(self, inputs):
        return self.model(inputs)

    @staticmethod
    def state_to_tensors(state):
        if type(state) != tuple:
            return BaseModel.state_to_tensors((state,))

        tensor_state = []
        for input in state:
            input = tf.convert_to_tensor(input)
            input = tf.expand_dims(input, 0)
            tensor_state.append(input)

        return tensor_state


class ActorCritic(BaseModel):
    def __init__(self, env):
        super().__init__(env)
        self.architecture_name = "A-C"
        self.critic_loss_function = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def create_heads(self, x):
        # unique policy network
        p = Dense(units=NUM_FEATURES, activation="relu", name="Policy-Features")(x)
        p = Dense(self.action_size, activation="softmax", name="Policy-Head")(p)

        # unique value network
        v = Dense(units=NUM_FEATURES, activation="relu", name="Value-Features")(x)
        v = Dense(1, activation="tanh", name="Value-Head")(v)

        return [p, v]

    def verbose_greedy_prediction(self, state):
        state = self.state_to_tensors(state)

        action_probs, value = self.model(state)
        greedy_action = np.argmax(action_probs)

        named_action_probs = named_array(action_probs)
        print("\n\nAction Policy:")
        print(named_action_probs)
        print("\nValue Prediction:", round(value.numpy()[0, 0], 3))

        return greedy_action

    def compute_loss(self, action_probs, values, rewards, entropies):
        returns = self.get_expected_return(rewards)
        advantage = np.array(returns) - np.array(values)

        action_log_probs = tf.math.log(action_probs)

        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = self.critic_loss_function(values, returns)

        return actor_loss + critic_loss - entropies

    def run_episode(self, state, exploration_rate):
        episode_action_probs = []
        episode_values = []
        episode_rewards = []
        episode_entropies = []
        for step in range(MAX_STEPS_EACH_EPISODE):
            state = self.state_to_tensors(state)

            action_probs, value = self.model(state)

            if np.random.rand() <= exploration_rate:
                action = np.random.choice(NUM_ACTIONS)
            else:
                action = np.random.choice(NUM_ACTIONS, p=np.squeeze(action_probs))

            state, reward, done, legal, info = self.env.step(action)

            action_log_probs = tf.math.log(action_probs)
            entropy = -tf.math.reduce_sum(action_probs * action_log_probs)

            episode_action_probs.append(action_probs[0, action] + self.eps)
            episode_values.append(value[0, 0])
            episode_rewards.append(reward)
            episode_entropies.append(entropy)

            if done or not legal:
                break

        loss = self.compute_loss(
            episode_action_probs, episode_values, episode_rewards, episode_entropies
        )
        episode_score = self.min_max_scaling(episode_rewards[-1])

        return loss, episode_score


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
        state = self.state_to_tensors(state)

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

    def run_episode(self, state, exploration_rate):
        episode_q_values = []
        episode_rewards = []
        for step in range(MAX_STEPS_EACH_EPISODE):
            state = self.state_to_tensors(state)
            q_values = self.model(state)

            if np.random.rand() <= exploration_rate:
                action = np.random.choice(NUM_ACTIONS)
            else:
                action = np.argmax(q_values)

            state, reward, done, legal, info = self.env.step(action)

            episode_q_values.append(q_values[0, action])
            episode_rewards.append(reward)

            if done or not legal:
                break

        loss = self.compute_loss(episode_q_values, episode_rewards)
        episode_score = self.min_max_scaling(episode_rewards[-1])

        return loss, episode_score
