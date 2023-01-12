import numpy as np
import tensorflow as tf
import os
import gym
from gym.envs.registration import register

from settings import GYM_ID


def set_default_options():
    np.set_printoptions(precision=4, suppress=True)

    np.random.seed(42)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)
    # suppress AVX_WARNING!
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    register_gym()


def register_gym():
    register(id=GYM_ID, entry_point="environment.profit_gym:ProfitGym")


def make_gym(width, height, field_of_vision):
    return gym.make(
        GYM_ID,
        width=width,
        height=height,
        field_of_vision=field_of_vision,
        turns=50,
        products={},
    )
