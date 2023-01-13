from buildings import *
import environment.environment as environment

from contextlib import contextmanager
import sys
import os
import json


def environment_from_json(filename):
    """loads a task from file (json format) and automatically creates the described environment with all its buildings, products and #turns

    Args:
        filename (str): path to the json-file

    Returns:
        Environment: environemnt object
    """
    with open(filename) as f:
        task = json.load(f)

    env = environment.Environment(
        task["width"], task["height"], task["turns"], task["products"]
    )
    for obj in task["objects"]:
        classname = obj["type"].capitalize()
        args = []
        args.append((obj["x"], obj["y"]))

        if "subtype" not in obj:
            args.append(0)
        else:
            args.append(obj["subtype"])
        if "width" in obj:
            args.extend([obj["width"], obj["height"]])

        building = globals()[classname](*args)

        if env.is_legal_position(building):
            env.add_building(building)
        else:
            print(f"UNABLE TO PLACE {classname} at {args[0]}\n")
    return env


def environment_to_json(env, filename):
    """parses an environment to a json file

    Args:
        env (Environment): the environment that shall be parsed to a json file
        filename (str): path to where the file should be stored
    """
    env_dict = {}
    env_dict["width"] = env.width
    env_dict["height"] = env.height
    env_dict["objects"] = []
    for building in env.buildings:
        env_dict["objects"].append(building.to_json())
    for obstacle in env.obstacles:
        env_dict["objects"].append(obstacle.to_json())
    env_dict["products"] = env.products
    env_dict["turns"] = env.turns
    env_dict["time"] = env.time

    path = os.path.join(".", "tasks", "solutions")
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(os.path.join(path, filename), "w") as jsonfile:
        json.dump(env_dict, jsonfile, separators=(",", ":"), indent=4)

    print(f"final solution has been saved to: {os.path.join(path, filename)}")


def environment_to_placeable_buildings_list(env):
    """parses only placable buildings of an environment to a json file.
    This is our final output for the Informaticup Profit Challenge.

    Args:
        env (Environment): the environment that shall be parsed to a json file
        filename (str): path to where the file should be stored
    """

    building_list = [
        building.to_json() for building in env.buildings if building.is_placeable()
    ]
    return json.dumps(building_list, indent=4)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
