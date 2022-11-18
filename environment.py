from typing import List

from shapes import *
from classes.buildings import *
import task_generator

from helper.dicts.placement_rules import *
from helper.constants.settings import *
from helper.functions.file_handler import *

import os
import random

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

        self.task_generator = task_generator.TaskGenerator(self)

        self.empty()

    def empty(self):
        self.buildings: List(Building) = []
        self.grid = np.full((self.height, self.width), " ")

    def add_building(self, building: Building):
        """Adds the individual tiles of a new building to the grid, provided that it has a valid position (see `Environment.is_legal_position`);

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            Building: returns building object or None if building could not be added
        """
        assert building not in self.buildings

        if not self.is_legal_position(building):
            return None

        for (tile_x, tile_y, element) in iter(building):
            self.grid[tile_y, tile_x] = element

        for other_building in self.buildings:
            if self.would_connect_to(other_building, building):
                other_building.add_connection(building)
            if self.would_connect_to(building, other_building):
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
        assert building in self.buildings

        for (tile_x, tile_y, element) in iter(building):
            self.grid[tile_y, tile_x] = " "

        for other_building in self.buildings:
            if building in other_building.connections:
                other_building.remove_connection(building)
        building.clear_connections()

        self.buildings.remove(building)

        return building

    def is_legal_position(self, building: Building):
        """Check whether a building that is not yet part of the enviornment has a valid position

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            bool: validity of the position
        """
        assert building not in self.buildings

        if self.is_out_off_bounds(building):
            return False
        if self.intersects_with_building(building):
            return False
        if self.violates_legal_connection(building):
            return False
        if self.violates_single_input(building):
            return False
        return True

    def coords_out_off_bounds(self, x, y):
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return True
        return False

    def is_tile_empty(self, x, y):
        if self.coords_out_off_bounds(x, y) or self.grid[y, x] != " ":
            return False
        return True

    def is_out_off_bounds(self, building):
        # iterate over non-empty elements of the building
        for (tile_x, tile_y, element) in iter(building):
            if self.coords_out_off_bounds(tile_x, tile_y):
                return True
        return False

    def get_coords_around_position(self, x, y):
        top = x, y - 1
        right = x + 1, y
        bottom = x, y + 1
        left = x - 1, y
        return (top, right, bottom, left)

    def get_adjacent_positions(self, positions, empty_only=False):
        """returns all coordinates adjacent to a list of positions.
        Use empty_only=True to filter out non-empty tiles.

        Args:
            positions (List((x, y)), or (x, y)): a list of multiple positions or a single (x, y) tuple
            empty_only (bool, optional): if True: only empty adjacent positions are returned. Defaults to False.

        Returns:
            List((x, y)): adjacent positions
        """
        if type(positions) == tuple:
            return self.get_adjacent_positions([positions])

        adjacent_positions = []
        for x, y in positions:
            for adjacent_x, adjacent_y in self.get_coords_around_position(x, y):
                if not empty_only or self.is_tile_empty(adjacent_x, adjacent_y):
                    adjacent_positions.append((adjacent_x, adjacent_y))
        return adjacent_positions

    def intersects_with_building(self, building):
        for (tile_x, tile_y, element) in iter(building):
            if not self.is_tile_empty(tile_x, tile_y):
                return True
        return False

    def violates_legal_connection(self, building):
        for other_building in self.buildings:
            if self.would_connect_to(building, other_building):
                if type(other_building) not in LEGAL_CONNECTIONS[type(building)]:
                    return True
            if self.would_connect_to(other_building, building):
                if type(building) not in LEGAL_CONNECTIONS[type(other_building)]:
                    return True
        return False

    def violates_single_input(self, building: Building):
        for out_x, out_y in building.get_output_positions():
            if len(self.get_adjacent_inputs(out_x, out_y)) > 1:
                return True

        for in_x, in_y in building.get_input_positions():
            for out_x, out_y in self.get_adjacent_outputs(in_x, in_y):
                # assume building has not been placed yet!
                assert building not in self.buildings
                if len(self.get_adjacent_inputs(out_x, out_y)) > 0:
                    return True
        return False

    def get_adjacent_inputs(self, x, y):
        return self.get_adjacent_elements(x, y, "+")

    def get_adjacent_outputs(self, x, y):
        return self.get_adjacent_elements(x, y, "-")

    def get_adjacent_elements(self, x, y, element):
        adjacent_positions = []
        for adjacent_x, adjacent_y in self.get_coords_around_position(x, y):
            if self.coords_out_off_bounds(adjacent_x, adjacent_y):
                continue
            if self.grid[adjacent_y, adjacent_x] == element:
                adjacent_positions.append((adjacent_x, adjacent_y))

        return adjacent_positions

    def get_min_distance(self, output_building: Building, input_building: Building):
        min_distance = None
        for out_x, out_y in output_building.get_output_positions():
            for in_x, in_y in input_building.get_input_positions():
                distance = self.get_tile_distance(out_x, out_y, in_x, in_y)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance

    def get_tile_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def would_connect_to(self, output_building, input_building):
        """tests whether output_building's outputs can connect to input_building's inputs.
        buildings need not to be part of the environment.

        Args:
            output_building (building):  building with outgoing connections
            input_building (building):  building with incoming connections

        Returns:
            bool: True iff buildings output/input are adjacent
        """
        return self.get_min_distance(output_building, input_building) == 1

    def is_connected(self, output_building, input_building):
        """tests whether output_building is connected to input_building via other buildings

        Args:
            output_building (building):  building with outgoing connections
            input_building (building):  building with incoming connections

        Returns:
            bool: True iff buildings are connected
        """

        for next_building in output_building.connections:
            if next_building == input_building:
                return True
            elif self.is_connected(next_building, input_building):
                return True
        return False

    def has_connection_loop(self, building_1, building_2):
        forward_connection = self.is_connected(building_1, building_2)
        backward_connection = self.is_connected(building_2, building_1)
        return forward_connection and backward_connection

    def __str__(self):
        """printable representation;
        displayed as a ASCII grid and possibly additional information like #turns, products, ...

        Returns:
            str: string repesentation of the environment
        """
        return f"\n{self.grid}\n".replace("'", "")


if __name__ == "__main__":
    # single input for each output test
    env = Environment(10, 10, 50, {})
    env.add_building(Deposit((2, 2), 0, 1, 1))
    assert env.add_building(Mine((2, 4), 1)) is not None
    assert env.add_building(Mine((4, 1), 0)) is None

    # load and display sample task
    filename = os.path.join(".", "tasks", "json_test", "minimal.json")
    env = environment_from_json(filename)

    print(env)
