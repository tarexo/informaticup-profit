import os
from environment import Environment
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh


def generate_solution(filename):
    env = fh.environment_from_json(filename)
    print(env.grid)

    add_factory(env)

    add_mine(env)


def add_factory(env):
    print(env.grid)


def add_mine(env):
    print(env.grid)


if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "002.task.json")
    generate_solution()
