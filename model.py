import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
from helper.constants.settings import *


def compile_model(model):
    optimizer = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="mean_squared_loss")


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(MAX_WIDTH, MAX_HEIGHT, 3))

    # adding the convolutional layers
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(
            filters=conv_size,
            kernel_size=3,
            padding="same",
            activation="relu",
            data_format="channels_last",
        )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, "relu")(x)
    actor = layers.Dense(64, activation="softmax")(x)
    critic = layers.Dense(1, "sigmoid")(x)

    return models.Model(inputs=board3d, outputs=[actor, critic])
