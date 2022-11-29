# File containing the various models that can be used

from keras.models import Model
from keras.layers import (
    Dense,
    Conv2D,
    BatchNormalization,
    Flatten,
    Input,
    Activation,
    Add,
)
from keras.activations import relu
from tensorflow.keras.layers import Layer

from helper.constants.settings import *


class AlphaZero:
    """The alphazero model."""

    def __init__(
        self,
        block_filters=256,
        block_kernel=3,
        blocks=10,
        policy_filters=2,
        value_filters=1,
        value_hidden=256,
    ):
        super(AlphaZero, self).__init__()
        board_size = (MAX_HEIGHT, MAX_WIDTH, NUM_CHANNELS)
        action_size = NUM_ACTIONS

        # initial conv block
        inp = Input(shape=board_size)
        x = Conv2D(block_filters, block_kernel, padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # residual blocks
        for i in range(blocks):
            # TODO: compress the residual block into a single layer
            res = Conv2D(block_filters, block_kernel, padding="same")(x)
            res = BatchNormalization()(res)
            res = Activation("relu")(res)
            res = Conv2D(block_filters, block_kernel, padding="same")(res)
            res = BatchNormalization()(res)
            res = Add()([res, x])
            x = Activation("relu")(res)

        # policy head
        p = Conv2D(policy_filters, kernel_size=1)(x)
        p = BatchNormalization()(p)
        p = Activation("relu")(p)
        p = Flatten()(p)
        p = Dense(action_size, activation="softmax", name="policy")(p)

        # value head
        v = Conv2D(value_filters, kernel_size=1)(x)
        v = BatchNormalization()(v)
        v = Activation("relu")(v)
        v = Flatten()(v)
        v = Dense(value_hidden, activation="relu")(v)
        v = Dense(1, activation="tanh", name="value")(v)

        self.model = Model(inputs=inp, outputs=[p, v])
