from shapes import *
from classes.buildings import *
from helper.dicts.convert_actions import *
from task_generator import TaskGenerator

import json
import os
import random

import gym
from gym import spaces

# create a nice output when displaying the entire grid
np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


class Environment(gym.Env):
    """
    Profit Game Environment
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, width, height, turns, products: dict, render_mode=None):
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

        self.task_generator = TaskGenerator(self)

        self.empty()
        self.setup_gym(render_mode)

    def setup_gym(self, render_mode):
        # [empty, obstacle, input, output] in a 100x100 grid
        self.observation_space = spaces.MultiBinary((4, 100, 100))

        # We have 16 different buildings (TODO: +4 for combiners) at four possible positions (at most 3 valid) adjacent to the input tile
        self.action_space = spaces.MultiDiscrete((16, 4))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # task generator modifies self (this environment!)
        deposit, factory = self.task_generator.generate_simple_task()
        self.current_building = deposit
        self.target_building = factory

        observation = self.grid_to_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        building_action, positional_action = action

        x, y = self.current_building.get_output_positions()[0]
        x_offset, y_offset = POSITIONAL_ACTION_TO_DIRECTION[positional_action]
        position = (x + x_offset, y + y_offset)

        BuildingClass, subtype = BUILDING_ACTION_TO_CLASS_SUBTYPE[building_action]
        new_building = BuildingClass(position, subtype)

        print(new_building)
        assert self.is_legal_position(new_building)
        self.add_building(new_building)

        # An episode is done iff the agent has reached the target
        terminated = self.is_connected(new_building, self.target_building)
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self.grid_to_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def render(self):
        print(self)

    def grid_to_observation(self):
        return self.grid

    def empty(self):
        self.buildings = []
        self.grid = np.full((self.height, self.width), " ")

    def add_building(self, building: Building):
        """Adds the individual tiles of a new building to the grid, provided that it has a valid position (see `Environment.is_legal_position`);

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            Building: returns building object or None if building could not be added
        """
        if not self.is_legal_position(building):
            return None

        # iterate over non-empty elements of the building shape
        for (tile_offset_x, tile_offset_y, element) in iter(building.shape):
            # calculate tile position on the grid relative to the center of the shape
            x = building.x + tile_offset_x
            y = building.y + tile_offset_y
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

        return building

    def remove_building(self, building):
        """Removes the individual tiles of a new building from the grid;

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            Building: returns removed building object
        """

        # iterate over non-empty elements of the building shape
        for (tile_offset_x, tile_offset_y, element) in iter(building.shape):
            # calculate tile position on the grid relative to the center of the shape
            x = building.x + tile_offset_x
            y = building.y + tile_offset_y
            self.grid[y, x] = " "

        # remove building connections
        for pos in self.get_adjacent_positions(building.get_input_positions()):
            for other_building in self.buildings:
                if pos in other_building.get_output_positions():
                    other_building.remove_connection(building)

        building.clear_connections()

        self.buildings.remove(building)

        return building

    def get_legal_building(self, BuildingClass, subtype):
        """suggests a random (but legal) building position.
        simple brute_force approach for now: pick a random position and test it's legality

        Args:
            BuildingClass (class): the class of the building that is supposed to be placed
            subtype (int): subtype of the building

        Returns:
            Building: a building object at random (but legal) position that has not yet been placed inside the environment

        Throws:
            MaxRecursionError
        """
        assert not issubclass(BuildingClass, UnplacableBuilding)

        x, y = random.randint(0, self.width), random.randint(0, self.height)
        building = BuildingClass((x, y), subtype)
        if not self.is_legal_position(building):
            return self.get_legal_building(BuildingClass, subtype)
        return building

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
            x = building.x + tile_offset_x
            y = building.y + tile_offset_y

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

    def remove_connected_buildings(self, output_building, input_building):
        if not self.is_connected(output_building, input_building):
            print(
                f"WARNING: unable to remove connection as {output_building} and {input_building} are not connected."
            )
            return None

        removed_buildings = []
        for next_building in output_building.connections:
            if next_building == input_building:
                return []
            if self.is_connected(next_building, input_building):
                connected_buildings = self.remove_connected_buildings(
                    next_building, input_building
                )
                self.remove_building(next_building)
                removed_buildings.append(next_building)
                removed_buildings.extend(connected_buildings)

        return removed_buildings

    def is_connected(self, output_building, input_building):
        for next_building in output_building.connections:
            if next_building == input_building or self.is_connected(
                next_building, input_building
            ):
                return True
        return False

    def distance(self, output_building: Building, input_building: Building):
        min_distance = None
        for out_x, out_y in output_building.get_output_positions():
            for in_x, in_y in input_building.get_input_positions():
                distance = abs(out_x - in_x) + abs(out_y - in_y)
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

    print(env)
