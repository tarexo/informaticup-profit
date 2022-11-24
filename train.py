from model import *
from gym_test import *
from helper.dicts.convert_actions import *

import tensorflow as tf

import numpy as np


def load_env(name):
    register_gym(name)
    return make_gym(name)


def train():
    env = load_env("Profit-v0")
    model = build_model(8, 4)
    compile_model(model)
    # model.summary()

    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    state = tf.transpose(state, [0, 2, 3, 1])

    action_prediction, value_prediction = model.predict(state)
    greedy_action = np.argmax(action_prediction)
    positional_action = greedy_action // 16
    building_action = greedy_action % 16

    action = (building_action, positional_action)
    env.render()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()


if __name__ == "__main__":
    train()
