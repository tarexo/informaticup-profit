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

        # add building connections
        for pos in self.get_adjacent_positions(building.get_input_positions()):
            for other_building in self.buildings:
                if pos in other_building.get_output_positions():
                    other_building.add_connection(building)

        for pos in self.get_adjacent_positions(building.get_output_positions()):
            for other_building in self.buildings:
                if pos in other_building.get_input_positions():
                    building.add_connection(other_building)

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

    def get_adjacent_positions(self, positions, empty_only=False):
        # works for single position as well as a list of positions
        if type(positions) == tuple:
            return self.get_adjacent_positions([positions])

        adjacent_positions = []
        for x, y in positions:
            for x_offset, y_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adjacent_x = x + x_offset
                adjacent_y = y + y_offset
                if not empty_only or self.is_tile_empty(adjacent_x, adjacent_y):
                    adjacent_positions.append((adjacent_x, adjacent_y))
        return adjacent_positions

    def is_connected(self, output_building: Building, input_building: Building):
        for next_building in output_building.connections:
            if next_building == input_building or self.is_connected(
                next_building, input_building
            ):
                return True
        return False

    def distance(self, output_building: Building, input_building: Building):
        min_distance = None
        for out_pos in output_building.get_output_positions():
            for in_pos in input_building.get_input_positions():
                distance = abs(out_pos[0] - in_pos[0]) + abs(out_pos[1] - in_pos[1])
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance

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
            else:
                print(f"UNABLE TO PLACE {obj['type']} at position {position}\n")
        return env

    @staticmethod
    def to_json(filename):
        """parses an environment to a json file

        Args:
            filename (str): path to where the file should be stored
        """
        # TODO add environment as argument
        pass

    def __repr__(self):
        """printable representation;
        displayed as a ASCII grid and possibly additional information like #turns, products, ...

        Returns:
            str: string repesentation of the environment
        """
        return f"\nCOMPLETE ENVIRONMENT:\n\n{self.grid}\n".replace("'", "")


if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "001.task.json")
    filename = os.path.join(".", "tasks", "manual solutions", "task_1.json")
    env = Environment.from_json(filename)

    # print(env)
