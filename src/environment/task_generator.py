from __future__ import annotations

import environment.environment
from buildings import *
from settings import *
from helper.placement_rules import *

import random


class TaskGenerator:
    """Manipulates a given envionment in order to automatically generate tasks."""

    def __init__(self, env: environment.Environment, seed=None):
        random.seed(seed)
        self.env = env

    def generate_task(self, difficulty):
        obstacle_probability, distance_range = self.get_difficulty_params(difficulty)

        self.env.empty()

        deposit = self.place_at_random_position(Deposit, 0)
        constraint = self.distance_constraint(distance_range, deposit)
        factory = self.place_at_random_position(Factory, 0, constraint)

        connections = self.connect_deposit_factory(deposit, factory)

        if random.random() < 0.25:
            other_deposit = self.place_at_random_position(Deposit, 0)
            other_factory = self.place_at_random_position(Factory, 0)
            false_targets = self.connect_deposit_factory(
                other_deposit, other_factory, can_fail=True
            )
            self.env.make_untargetable(false_targets + [other_factory])

        other_connections = []
        if random.random() < 0.35:
            another_deposit = self.place_at_random_position(Deposit, 0)
            other_connections = self.connect_deposit_factory(
                another_deposit, factory, can_fail=True
            )

        if not NO_OBSTACLES:
            self.add_obstacles(p=obstacle_probability)

        assert len(connections) >= 1
        start_building = mine = connections[0]

        for building in connections[1:]:
            self.env.remove_building(building)
            if other_connections:
                if not self.env.is_connected(other_connections[-1], factory):
                    self.env.add_building(building)

        while self.env.is_connected(start_building, factory):
            return self.generate_task(difficulty)

        return start_building, factory

    def get_difficulty_params(self, difficulty):
        obstacle_probability = MAX_OBSTACLE_PROBABILITY * difficulty
        max_distance = 7 + int((self.env.width + self.env.height) * difficulty)
        if SIMPLE_GAME:
            distance_range = range(3, max_distance, 2)
        else:
            distance_range = range(6, max_distance)

        return obstacle_probability, distance_range

    def connect_deposit_factory(self, deposit, factory, can_fail=False):
        connections = []
        new_building = deposit
        while not self.env.is_connected(new_building, factory):
            best_buildings = self.get_best_buildings(new_building, factory)
            if not best_buildings:
                if can_fail:
                    for building in connections:
                        self.env.remove_building(building)
                    return []
                print(self.env)
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
                    self.env.add_building(obstacle, force=True)

    def place_at_random_position(self, BuildingClass, subtype, constraint=None):
        building = self.get_random_legal_building(BuildingClass, subtype)
        if constraint and not constraint(building):
            return self.place_at_random_position(BuildingClass, subtype, constraint)
        self.env.add_building(building)
        return building

    def place_obstacle_in_middle(self):
        obstacle = Obstacle((self.env.width // 2, self.env.height // 2), 0, 1, 1)
        return self.env.add_building(obstacle)

    def get_random_legal_building(
        self, BuildingClass, subtype, width=None, height=None
    ):
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
                if BuildingClass == Factory or BuildingClass == SimpleFactory:
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

        # if not best_buildings:
        #     print(self.env)
        #     print(start_building.get_output_positions())
        #     print(target_building.get_input_positions())

        return best_buildings

    def distance_constraint(self, distances, other_building):
        if distances is None:
            return lambda building: True
        return (
            lambda building: self.env.get_min_distance(other_building, building)
            in distances
        )
