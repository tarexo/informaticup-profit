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
        if not self.is_legal_position(building):
            return None

        # iterate over non-empty elements of the building shape
        for (x, y, element) in iter(building):
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

    def remove_connecting_buildings(self, output_building, input_building):
        """Removes all buildings connecting output_building to input_building.
        output_building and input_building will be kept.

        Args:
            output_building (building): building with outgoing connections
            input_building (building): building with incoming connections

        Returns:
            building[]: a list of all removed buildings
        """
        assert self.is_connected(output_building, input_building)

        removed_buildings = []
        for building in output_building.connections:
            if self.is_connected(building, input_building):
                branch = self.remove_connecting_buildings(building, input_building)
                removed_buildings.append(building)
                removed_buildings.extend(branch)
                self.remove_building(building)

        return removed_buildings

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
        """Check whether a building has a valid position:

        Args:
            building (Building): Factory, Deposit, Obstacle, ...

        Returns:
            bool: validity of the position
        """

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
        if self.grid[y, x] != " ":
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
        # works for single position (x, y) as well as a list of positions [(x_1, y_1), ... (x_n, y_n)]
        if type(positions) == tuple:
            return self.get_adjacent_positions([positions])

        adjacent_positions = []
        for x, y in positions:
            for adjacent_x, adjacent_y in self.get_coords_around_position(x, y):
                if not empty_only or self.is_tile_empty(adjacent_x, adjacent_y):
                    adjacent_positions.append((adjacent_x, adjacent_y))
        return adjacent_positions

    def intersects_with_building(self, building):
        # iterate over non-empty elements of the building
        for (tile_x, tile_y, element) in iter(building):
            if self.coords_out_off_bounds(tile_x, tile_y):
                return True
            if not self.is_tile_empty(tile_x, tile_y):
                return True
        return False

    def violates_legal_connection(self, building):
        for other_building in self.buildings:
            if type(other_building) not in LEGAL_CONNECTIONS[type(building)]:
                if self.would_connect_to(building, other_building):
                    return True
            if type(building) not in LEGAL_CONNECTIONS[type(other_building)]:
                if self.would_connect_to(other_building, building):
                    return True
        return False

    def violates_single_input(self, building: Building):
        for out_x, out_y in building.get_output_positions():
            if len(self.get_adjacent_inputs(out_x, out_y)) > 1:
                return True

        for in_x, in_y in building.get_input_positions():
            for out_x, out_y in self.get_adjacent_outputs(in_x, in_y):
                # assume building has not been placed yet!
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

    def has_connection_loop(self, building_1, building_2):
        forward_connection = self.is_connected(building_1, building_2)
        backward_connection = self.is_connected(building_2, building_1)
        return forward_connection and backward_connection

    def is_connected(self, output_building, input_building):
        """tests whether output_building is connected to input_building via other buildings

        Args:
            output_building (building):  building with outgoing connections
            input_building (building):  building with incoming connections

        Returns:
            bool: True iff buildings are connected
        """

        for next_building in output_building.connections:
            if next_building == input_building or self.is_connected(
                next_building, input_building
            ):
                return True
        return False

    def get_tile_distance(self, x_1, y_1, x_2, y_2):
        return abs(x_1 - x_2) + abs(y_1 - y_2)

    def get_min_distance(self, output_building: Building, input_building: Building):
        min_distance = None
        for out_x, out_y in output_building.get_output_positions():
            for in_x, in_y in input_building.get_input_positions():
                distance = abs(out_x - in_x) + abs(out_y - in_y)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance

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
    filename = os.path.join(".", "tasks", "manual solutions", "task_1.json")
    env = environment_from_json(filename)

    print(env)
