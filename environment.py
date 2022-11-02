from shapes import *
from classes.buildings import *

from dataclasses import dataclass
import json
import os

# create a nice output when displaying the entire grid
np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


class Environment:
    """
    Profit Game Environment
    """

    def __init__(self, width, height, turns, products: dict):
        """initialize environment

        Args:
            width (int): width of the grid
            height (int): height of the grid
            turns (int): maximum number of turns
            products (dict): dictonary of all possible products; each with a resource recipe and the #points it is worth
        """
        self.width = width
        self.height = height
        self.turns = turns
        self.products = products

        self.buildings = []
        self.grid = np.full((height, width), " ")

    def add_building(self, building: Building):
        """Adds the individual tiles of a new building to the grid, provided that it has a valid position (see `Environment.is_legal_position`);

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            bool: success
        """
        if not self.is_legal_position(building):
            return False

        # iterate over non-empty elements of the building shape
        for (tile_offset_x, tile_offset_y, element) in iter(building.shape):
            # calculate tile position on the grid relative to the center of the shape
            x = building.position[0] + tile_offset_x
            y = building.position[1] + tile_offset_y
            self.grid[y, x] = element

        self.buildings.append(building)

        return True

    def is_legal_position(self, building: Building):
        """Check wether a building has a valid position:
        --> is inside grid bounds?
        --> no intersection with other buildings?
        --> TODO no violation of constraints?

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            bool: validity of the position
        """

        # iterate over non-empty elements of the building's shape
        for (tile_offset_x, tile_offset_y, element) in iter(building.shape):
            # calculate tile position on the grid relative to the center of the building
            x = building.position[0] + tile_offset_x
            y = building.position[1] + tile_offset_y

            # check whether each individual element can be placed on an empty tile
            if not self.is_tile_empty(x, y):
                return False

                # ToDo: check_constraints (mines cannot be placed next to other mines)
        return True

    def is_tile_empty(self, x, y):
        """Checks wether a tile is empty
        (!) returns False if position is out of grid bounds

        Args:
            x (int): x grid coordinate
            y (int): y grid coordinate

        Returns:
            bool: empty_tile?
        """
        # check whether tile is out of grid bounds
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False

        # check for other buildings
        if self.grid[y, x] != " ":
            return False
        return True

    @staticmethod
    def from_json(filename):
        """loads a task from file (json format) and automatically creates the described environment with all its buildings, products and #turns

        Args:
            filename (str): path to the json-file

        Returns:
            Environment: environemnt object
        """
        with open(filename) as f:
            task = json.load(f)

        env = Environment(
            task["width"], task["height"], task["turns"], task["products"]
        )
        for obj in task["objects"]:
            position = (obj["x"], obj["y"])
            if obj["type"] == "obstacle":
                building = Obstacle(position, obj["width"], obj["height"])
            elif obj["type"] == "deposit":
                building = Deposit(
                    position, obj["width"], obj["height"], obj["subtype"]
                )
            elif obj["type"] == "mine":
                building = Mine(position, obj["subtype"])
            elif obj["type"] == "combiner":
                building = Combiner(position, obj["subtype"])
            elif obj["type"] == "factory":
                building = Factory(position, obj["subtype"])
            elif obj["type"] == "conveyor":
                building = Conveyor(position, obj["subtype"])
            else:
                print(f"UNKNOWN BUILDING TYPE: {obj['type']}\n")
                continue

            if env.is_legal_position(building):
                env.add_building(building)
                print(building)
                print()
            else:
                print(f"UNABLE TO PLACE {obj['type']} at position {position}\n")
        return env

    @staticmethod
    def to_json(env, filename):
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
        env_dict["turns"] = env.turns
        env_dict["form"] = {"products": env.products}

        with open(filename, "w") as jsonfile:
            json.dump(env_dict, jsonfile)

    def __repr__(self):
        """printable representation;
        displayed as a ASCII grid and possibly additional information like #turns, products, ...

        Returns:
            str: string repesentation of the environment
        """
        return f"\nCOMPLETE ENVIRONMENT:\n\n{self.grid}\n".replace("'", "")


if __name__ == "__main__":
    # filename = os.path.join(".", "tasks", "001.task.json")
    filename = os.path.join(".", "tasks", "manual solutions", "task_1.json")
    env = Environment.from_json(filename)

    out_filename = os.path.join(".", "tasks", "json_test", "task_1.json")
    Environment.to_json(env, out_filename)

    print(env)
