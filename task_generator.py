from environment import Environment
from classes.buildings import *

from helper.dicts.building_shapes import BUILDING_SHAPES

import random


class TaskGenerator:
    def __init__(self, env: Environment):
        self.env = env

    def generate_simple_task(self, save=False, output=False):
        self.env.empty()
        deposit = self.env.add_building(Deposit(2, 2, 0, 3, 3))
        factory = task_gen.place_building_at_random_position(Factory, 0)

        self.connect_deposit_factory(deposit, factory)
        if output:
            print(self.env)

    def connect_deposit_factory(self, deposit, factory):
        new_building = deposit
        while not env.is_connected(new_building, factory):
            best_buildings = self.get_best_buildings(new_building, factory)
            new_building = random.choice(best_buildings)
            env.add_building(new_building)

    def place_building_at_random_position(self, BuildingClass, subtype):
        building = self.env.get_legal_building(BuildingClass, subtype)
        self.env.add_building(building)
        return building

    def get_best_buildings(self, start_building: Building, goal_building: Building):
        """Finds the most suitable building(s) to place adjacent to the start_building
        in order to reduce the connection gap between the start and goal building
        according to the minimum distance heuristic between the new building's output and the goal_building's input.

        Args:
            start_building (Building): best_building(s) should fit next to the start_building's output
            goal_building (Building): best_building(s) should have minimum distance to one of the goal_building's inputs.

        Returns:
            best_buildings ([Building]): a list of equally suitable buildings
        """

        min_distance = None
        best_buildings = []

        out_positions = start_building.get_output_positions()
        for x, y in self.env.get_adjacent_positions(out_positions, empty_only=True):
            for BuildingClass in self.get_allowed_building_classes(start_building):
                for subtype in range(BuildingClass.NUM_SUBTYPES):
                    building = BuildingClass.from_input_position(x, y, subtype)
                    if self.env.is_legal_position(building):
                        distance = env.distance(building, goal_building)
                        if min_distance is None or distance <= min_distance:
                            if min_distance and distance < min_distance:
                                best_buildings = []
                            min_distance = distance
                            best_buildings.append(building)

        return best_buildings

    @staticmethod
    def get_allowed_building_classes(output_building):
        if type(output_building) == Deposit:
            return [Mine]
        elif type(output_building) == Mine:
            return [Conveyor, Combiner]
        else:
            return [Mine, Conveyor, Combiner]


if __name__ == "__main__":
    env = Environment(
        30,
        30,
        100,
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

    for i in range(10):
        task_gen.generate_simple_task(output=True)
