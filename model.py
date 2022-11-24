import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizer
from helper.constants.settings import *


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(3, MAX_WIDTH, MAX_HEIGHT))

    # adding the convolutional layers
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(
            filters=conv_size,
            kernel_size=3,
            padding="same",
            activation="relu",
            data_format="channels_first",
        )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, "relu")(x)
    critic = layers.Dense(1, "sigmoid")(x)
    actor = layers.Dense(64, activation="softmax")(x)

    return models.Model(inputs=board3d, outputs=[actor, critic])
