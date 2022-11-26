import tensorflow as tf
from tensorflow.keras import layers

from helper.constants.settings import *


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self):
        """Initialize."""
        super().__init__()

        self.common_hidden_layers = []
        for i in range(COMMON_CONV_DEPTH):
            # switch betweeen stride of 1 and stride of 2
            stride = 1 + ((i + 1) % 2)
            conv_layer = layers.Conv2D(
                filters=COMMON_CONV_SIZE,
                kernel_size=3,
                strides=(stride, stride),
                padding="same",
                activation="relu",
                data_format="channels_last",
            )
            self.common_hidden_layers.append(conv_layer)

        flattened_layer = layers.Flatten()
        self.common_hidden_layers.append(flattened_layer)

        for _ in range(COMMON_DENSE_DEPTH):
            dense_layer = layers.Dense(COMMON_DENSE_SIZE, activation="relu")
            self.common_hidden_layers.append(dense_layer)

        self.critic_hidden = layers.Dense(UNIQUE_DENSE_SIZE, activation="relu")
        self.actor_hidden = layers.Dense(UNIQUE_DENSE_SIZE, activation="relu")

        self.actor = layers.Dense(NUM_ACTIONS)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.common_hidden_layers:
            x = layer(x)
        actor_x = self.actor_hidden(x)
        critic_x = self.critic_hidden(x)
        return self.actor(actor_x), self.critic(critic_x)
