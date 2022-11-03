from environment import Environment
from classes.buildings import *

from helper.dicts.building_shapes import BUILDING_SHAPES

import random


def best_building(env: Environment, start_building: Building, goal_building: Building):
    allowed_building_classes = get_allowed_building_classes(start_building)

    min_distance = None
    best_buildings = []
    out_positions = start_building.get_output_positions()
    for input_position in env.get_adjacent_positions(out_positions, empty_only=True):
        for BuildingClass in allowed_building_classes:
            for subtype in range(BuildingClass.NUM_SUBTYPES):
                dummy_building = BuildingClass((0, 0), subtype)
                # combiners have 3 different inputs!
                for input_offset in dummy_building.get_input_positions():
                    # candidate_position = where to place building of certain subtype so that its input is adjacent to the deposits output
                    candidate_position = (
                        input_position[0] - input_offset[0],
                        input_position[1] - input_offset[1],
                    )
                    building = BuildingClass(candidate_position, subtype)

                    if env.is_legal_position(building):
                        distance = env.distance(building, goal_building)
                        if min_distance is None or distance == min_distance:
                            min_distance = distance
                            best_buildings.append(building)
                        elif distance < min_distance:
                            min_distance = distance
                            best_buildings = [building]

    if len(best_buildings) == 0:
        print(env)
    return random.choice(best_buildings)


def get_allowed_building_classes(output_building):
    if type(output_building) == Deposit:
        return [Mine]
    elif type(output_building) == Mine:
        return [Conveyor, Combiner]
    else:
        return [Mine, Conveyor, Combiner]


def connect_deposit_factory(env, deposit, factory):
    new_building = best_building(env, deposit, factory)
    env.add_building(new_building)
    while not env.is_connected(deposit, factory):
        new_building = best_building(env, new_building, factory)
        env.add_building(new_building)
    print(env)


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

    deposit = Deposit((2, 2), 3, 3, 0)
    factory = Factory((env.width - 7, env.height - 7), 0)

    env.add_building(deposit)
    env.add_building(factory)

    connect_deposit_factory(env, deposit, factory)
