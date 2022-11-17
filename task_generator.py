from __future__ import annotations

import environment
from classes.buildings import *
from helper.constants.settings import *
from helper.dicts.placement_rules import *

import random


class TaskGenerator:
    """Manipulates a given envionment in order to automatically generate tasks."""

    def __init__(self, env: environment.Environment):
        self.env = env

    def generate_simple_task(self, seed=None, save=False):
        random.seed(seed)

        self.env.empty()
        deposit = self.env.add_building(Deposit((2, 2), 0, 3, 3))
        factory = self.place_building_at_random_position(Factory, 0)

        connections = self.connect_deposit_factory(deposit, factory)
        self.add_obstacles(p=0.05)

        assert len(connections) >= 1
        mine = connections[0]
        self.remove_connecting_buildings(mine, factory)

        return mine, factory

    def connect_deposit_factory(self, deposit: Building, factory: Building):
        connections = []
        new_building = deposit
        while not self.env.is_connected(new_building, factory):
            best_buildings = self.get_best_buildings(new_building, factory)
            assert best_buildings
            new_building = random.choice(best_buildings)
            self.env.add_building(new_building)
            connections.append(new_building)

        return connections

    def add_obstacles(self, p=0.1):
        """adds 1x1 obstacles at empty tiles with some probability p

        Args:
            p (float, optional): probability for turning empty tile into obstacle. Defaults to 0.1.
        """
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.is_tile_empty(x, y) and random.random() < p:
                    obstacle = Obstacle((x, y), 0, 1, 1)
                    self.env.add_building(obstacle)

    def place_building_at_random_position(self, BuildingClass, subtype):
        building = self.get_random_legal_building(BuildingClass, subtype)
        self.env.add_building(building)
        return building

    def get_random_legal_building(self, BuildingClass, subtype):
        """suggests a random (but legal) building position.
        simple brute_force approach for now: pick a random position and test its legality

        Args:
            BuildingClass (class): the class of the building that is supposed to be placed
            subtype (int): subtype of the building

        Returns:
            Building: a building object at random (but legal) position that has not yet been placed inside the environment

        Throws:
            MaxRecursionError
        """
        assert not issubclass(BuildingClass, UnplacableBuilding)

        x, y = random.randint(0, self.env.width), random.randint(0, self.env.height)
        building = BuildingClass((x, y), subtype)
        if not self.env.is_legal_position(building):
            return self.get_random_legal_building(BuildingClass, subtype)
        return building

    def remove_connecting_buildings(self, output_building, input_building):
        """Removes all buildings connecting the output_building to the input_building.
        output_building and input_building will be kept.

        Args:
            output_building (building): building with outgoing connections
            input_building (building): building with incoming connections

        Returns:
            List(building): a list of all removed buildings
        """
        assert self.env.is_connected(output_building, input_building)

        removed_buildings = []
        for building in output_building.connections:
            if self.env.is_connected(building, input_building):
                branch = self.remove_connecting_buildings(building, input_building)
                removed_buildings.append(building)
                removed_buildings.extend(branch)
                self.env.remove_building(building)

        return removed_buildings

    def get_best_buildings(self, start_building: Building, target_building: Building):
        """Finds the most suitable building(s) to place adjacent to the start_building
        in order to reduce the connection gap between the start and target building
        according to the minimum distance heuristic between the new building's output and the target_building's input.

        Args:
            start_building (Building): best_building(s) should fit next to the start_building's output
            target_building (Building): best_building(s) should have minimum distance to one of the target_building's inputs.

        Returns:
            best_buildings ([Building]): a list of equally suitable buildings
        """

        min_distance = None
        best_buildings = []

        out_positions = start_building.get_output_positions()
        for x, y in self.env.get_adjacent_positions(out_positions, empty_only=True):
            for BuildingClass in LEGAL_CONNECTIONS[type(start_building)]:
                if BuildingClass == Factory:
                    continue
                for subtype in range(BuildingClass.NUM_SUBTYPES):
                    building = BuildingClass.from_input_position(x, y, subtype)
                    if self.env.is_legal_position(building):
                        distance = self.env.get_min_distance(building, target_building)
                        if min_distance is None or distance <= min_distance:
                            if min_distance and distance < min_distance:
                                best_buildings = []
                            min_distance = distance
                            best_buildings.append(building)

        return best_buildings


if __name__ == "__main__":
    env = environment.Environment(
        MAX_WIDTH,
        MAX_HEIGHT,
        50,
        [
            {
                "type": "product",
                "subtype": 0,
                "resources": [1, 0, 0, 0, 0, 0, 0, 0],
                "points": 10,
            }
        ],
    )

    task_gen = TaskGenerator(env)
    task_gen.generate_simple_task()
    print(task_gen.env)