import os
from environment import Environment
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh


def generate_solution(filename):
    env = fh.environment_from_json(filename)
    add_factory(env)
    add_mine(env)
    find_path(env)


def add_factory(env: Environment):
    for i in range(len(env.grid) - 1):
        for j in range(len(env.grid[i]) - 1):
            factory = buildings.Factory((j, i), 0)
            success = env.add_building(factory)
            if success == factory:
                return


def add_mine(env: Environment):
    deposit = None
    for building in env.buildings:
        if type(building) == buildings.Deposit:
            deposit = building
    if deposit == None:
        return
    x = deposit.x
    y = deposit.y
    width = deposit.width
    height = deposit.height
    lower = []
    upper = []
    left = []
    right = []

    for i in range(y, y + height):
        if x - 2 >= 0:
            left.appen([x - 2, i])
        if x + width + 1 < env.grid.shape[1]:
            right.append([x + width + 1, i])

    for i in range(x, x + width):
        if y - 2 >= 0:
            upper.appen([i, y - 2])
        if y + height + 1 < env.grid.shape[0]:
            lower.append([i, y + height + 1])

    positions = [right, lower, left, upper]
    for i in range(len(positions)):
        for pos in positions[i]:
            mine = buildings.Mine((pos[0], pos[1]), i)
            success = env.add_building(mine)
            if success == mine:
                return


def find_path(env: Environment):
    print(env)


if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "002.task.json")
    generate_solution(filename)
