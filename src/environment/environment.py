import environment.task_generator as task_generator
from .shapes import *
from buildings import *

from helper.placement_rules import *
from helper.file_handler import *
from settings import *

import random
import json

# create a nice output when displaying the entire grid
np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


class Environment:
    """
    Profit Game Environment
    """

    def __init__(self, width, height, turns, products, time=120):
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
        self.time = time

        self.task_generator = task_generator.TaskGenerator(self, seed=42)

        self.empty()

    def empty(self):
        self.buildings = []
        self.obstacles = []
        self.grid = np.full((self.height, self.width), " ")

    def reset_resources(self):
        for building in self.buildings:
            building.reset_resources()

    def add_building(self, building, force=False):
        """Adds the individual tiles of a new building to the grid, provided that it has a valid position (see `Environment.is_legal_position`);

        Args:
            building (Building): Factory, Deposit, Obstacle, ...
            force: only use this option if you are certain that it is a legal position (reduce computation)

        Returns:
            Building: returns building object or None if building could not be added
        """
        assert building not in self.buildings

        if not force and not self.is_legal_position(building):
            return None

        for (tile_x, tile_y, element) in iter(building):
            if self.grid[tile_y, tile_x] == " ":
                self.grid[tile_y, tile_x] = element
            else:
                # conveyor tunnel
                self.grid[tile_y, tile_x] = "O"

        if type(building) == Obstacle:
            self.obstacles.append(building)
            return building

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
        assert building in self.buildings or building in self.obstacles

        for (tile_x, tile_y, element) in iter(building):
            if self.grid[tile_y, tile_x] == "O":
                self.grid[tile_y, tile_x] = "<"
            else:
                self.grid[tile_y, tile_x] = " "

        if type(building) == Obstacle:
            self.obstacles.remove(building)
            return building

        for other_building in self.buildings:
            if building in other_building.connections:
                other_building.remove_connection(building)
        building.clear_connections()

        self.buildings.remove(building)

        return building

    def add_buildings(self, buildings, force=False):
        for building in buildings:
            self.add_building(building, force=force)

    def is_legal_position(self, building):
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
        # technically not illegal, but very dumb move!
        if self.creates_connection_loop(building):
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
                if self.conveyor_tunneling(tile_x, tile_y, element):
                    continue
                return True
        return False

    def conveyor_tunneling(self, tile_x, tile_y, element):
        if element in ["<", ">", "^", "v"]:
            if not self.coords_out_off_bounds(tile_x, tile_y):
                if self.grid[tile_y, tile_x] in ["<", ">", "^", "v"]:
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

    def violates_single_input(self, building):
        outgoing_connections = 0
        for other_building in self.buildings:
            if self.would_connect_to(building, other_building):
                outgoing_connections += 1
            elif self.would_connect_to(other_building, building):
                for connection in other_building.connections:
                    if self.is_diagonal_input(connection, building):
                        return True
                    elif self.is_opposite_input(connection, building):
                        return True

        if outgoing_connections > 1:
            return True
        return False

    def is_diagonal_input(self, building1, building2):
        inp1 = building1.get_input_positions()[0]
        inp2 = building2.get_input_positions()[0]

        x_diff, y_diff = inp1 - inp2
        return True if abs(x_diff) == 1 and abs(y_diff) == 1 else False

    def is_opposite_input(self, building1, building2):
        input1 = building1.get_input_positions()
        input2 = building2.get_input_positions()

        for inp1 in input1:
            for inp2 in input2:
                x_diff, y_diff = diff = inp1 - inp2

                if abs(x_diff) == 2 and abs(y_diff) == 0:
                    x, y = inp1 - (diff // 2)
                    if self.grid[(y, x)] == "-":
                        return True
                elif abs(x_diff) == 0 and abs(y_diff) == 2:
                    x, y = inp1 - (diff // 2)
                    if self.grid[(y, x)] == "-":
                        return True
        return False

    def creates_connection_loop(self, building):
        for other_building in self.buildings:
            if self.would_connect_to(building, other_building):
                if self.would_connect_to(other_building, building):
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

    def get_min_distance(self, output_building, input_building):
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

    def is_connected(self, output_building, input_building, visited_buildings=[]):
        """tests whether output_building is connected to input_building via other buildings
        Automatically checks for connection loops

        Args:
            output_building (building):  building with outgoing connections
            input_building (building):  building with incoming connections

        Returns:
            bool: True iff buildings are connected
        """

        for next_building in output_building.connections:
            if next_building == input_building:
                return True
            elif next_building not in visited_buildings and self.is_connected(
                next_building, input_building, visited_buildings + [output_building]
            ):
                return True
        return False

    def get_deposits(self, subtype=None):
        return self.get_all_building_types(Deposit, subtype)

    def get_factories(self, subtype=None):
        return self.get_all_building_types(Factory, subtype)

    def get_all_building_types(self, building_cls, subtype=None):
        if subtype is None:
            return [b for b in self.buildings if type(b) == building_cls]
        return [
            b
            for b in self.buildings
            if type(b) == building_cls and b.subtype == subtype
        ]

    def get_possible_factories(self, subtype, max=10):
        factories = []
        for y in range(self.height):
            for x in range(self.width):
                factory = Factory((x, y), subtype)
                if self.is_legal_position(factory):
                    factories.append(factory)

        random.shuffle(factories)
        return factories[:max]

    def get_possible_mines(self, deposit, factory=None, max=10):
        mines = []

        out_positions = deposit.get_output_positions()
        for x, y in self.get_adjacent_positions(out_positions, empty_only=True):
            for BuildingClass in LEGAL_CONNECTIONS[type(deposit)]:
                for subtype in range(BuildingClass.NUM_SUBTYPES):
                    building = BuildingClass.from_input_position(x, y, subtype)
                    if self.is_legal_position(building):
                        mines.append(building)

        random.shuffle(mines)
        mines = mines[:max]
        if factory:
            mines = sorted(mines, key=lambda m: self.get_min_distance(m, factory))
        return mines

    def make_untargetable(self, false_targets):
        for false_target in false_targets:
            for x, y in false_target.get_input_positions():
                self.grid[(y, x)] = "#"

    def make_targetable(self, true_targets):
        for true_target in true_targets:
            for x, y in true_target.get_input_positions():
                self.grid[(y, x)] = "+"

    def from_json(self, filename):
        with open(filename) as f:
            task = json.load(f)

        self.width = task["width"]
        self.height = task["height"]
        self.turns = task["turns"]
        self.products = task["products"]
        if "time" in task:
            self.time = task["time"]

        self.empty()

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
            self.add_building(building, force=True)

    def __str__(self):
        """printable representation;
        displayed as a ASCII grid and possibly additional information like #turns, products, ...

        Returns:
            str: string repesentation of the environment
        """
        return f"\n{self.grid}\n".replace("'", "")
