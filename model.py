import tensorflow as tf
from tensorflow.keras import layers

from helper.constants.settings import *


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, conv_size, conv_depth, dense_size, dense_depth):
        """Initialize."""
        super().__init__()

        self.hidden_layers = []
        for _ in range(conv_depth):
            conv_layer = layers.Conv2D(
                filters=conv_size,
                kernel_size=3,
                padding="same",
                activation="relu",
                data_format="channels_last",
            )
            self.hidden_layers.append(conv_layer)

        flattened_layer = layers.Flatten()
        self.hidden_layers.append(flattened_layer)

        for _ in range(dense_depth):
            dense_layer = layers.Dense(dense_size, activation="relu")
            self.hidden_layers.append(dense_layer)

        self.actor = layers.Dense(NUM_ACTIONS)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.actor(x), self.critic(x)
