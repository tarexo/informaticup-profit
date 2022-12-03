import os
from environment import Environment
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh


def generate_solution(filename):
    env = fh.environment_from_json(filename)

    add_factory(env)
    print(env.grid)
    add_mine(env)
    print(env.grid)


def add_factory(env: Environment):
    for i in range(len(env.grid) - 1):
        for j in range(len(env.grid[i]) - 1):
            factory = buildings.Factory((j, i), 0)
            success = env.add_building(factory)
            if success == factory:
                return


def add_mine(env):
    deposit = None
    for building in env.buildings:
        if type(building) == buildings.Deposit:
            deposit = building
    if deposit == None:
        return


if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "002.task.json")
    generate_solution(filename)
