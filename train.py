from model import *
from gym_test import *

import numpy as np


def load_env(name):
    register_gym(name)
    return make_gym(name)


def train():
    env = load_env("Profit-v0")
    model = build_model(8, 4)
    compile_model(model)
    model.summary()

    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)

    prediction = model.predict(state)
    print(prediction)


if __name__ == "__main__":
    train()
