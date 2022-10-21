from shapes import *

from dataclasses import dataclass
import json
import os


class Environment:
    def __init__(self, width, height, turns, products: dict):
        self.width = width
        self.height = height
        self.turns = turns
        self.products = products

        self.buildings = []

    def add_building(self, building: Building):
        if self.is_legal_position(building):
            self.buildings.append(building)
            return True
        return False

    def is_out_of_bounds(self, building: Building):
        # calculate building position relative to the entire shape
        shape_x = building.position[0] - building.shape.center[0]
        shape_y = building.position[1] - building.shape.center[1]
        
        height, width = building.shape.elements.shape
        for y_offset in range(height):
            for x_offset in range(width):
                if building.shape.elements[y_offset, x_offset] == 0:  # free space
                    continue
                y = shape_y + y_offset
                x = shape_x + x_offset
                if y < 0 or y >= self.height or x < 0 or x >= self.width:
                    return True
        return False

    def is_legal_position(self, building: Building):
        if self.is_out_of_bounds(building):
            return False
        # ToDo:
        # check_building_collisions
        # check_constraints (mines cannot be placed next to other mines)
        return True

    @staticmethod
    def from_json(filename):
        with open(filename) as f:
            task = json.load(f)

        env = Environment(
            task["width"], task["height"], task["turns"], task["products"]
        )
        for obj in task["objects"]:
            position = (obj["x"], obj["y"])
            if obj["type"] == "obstacle":
                building = Obstacle(position)  # ToDo: width, height
            elif obj["type"] == "deposit":
                building = Deposit(position, obj["subtype"])
            elif obj["type"] == "factory":
                building = Factory(position, obj["subtype"])
            elif obj["type"] == "conveyor":
                building = Conveyor(position, obj["subtype"])
            else:
                print(f"UNKNOWN BUILDING: {obj['type']}\n")

            env.add_building(building)
            print(building)
            print()
        return env

    @staticmethod
    def to_json(filename):
        pass

    def __repr__(self):
        pass
        # ToDo: nice presentation of the entire grid


class Obstacle(Building):
    def __init__(self, position):
        super().__init__(position, OBSTACLE_SHAPE)


class Deposit(Building):
    def __init__(self, position, subtype):
        super().__init__(position, DEPOSIT_SHAPE)

        # width/height should be adjustable in real game
        self.width = 3
        self.height = 3

        self.subtype = subtype
        self.resources[subtype] = self.width * self.height * 5


class Factory(Building):
    def __init__(self, position, subtype):
        super().__init__(position, FACTORY_SHAPE)

        self.subtype = subtype

        # product_recipe, points --> should be managed by environment or factory class?!


class Conveyor(Building):
    def __init__(self, position, subtype):
        self.subtype = subtype
        if subtype == 0:
            shape = CONVEYOR_SHAPE_0
        elif subtype == 1:
            shape = CONVEYOR_SHAPE_1
        elif subtype == 2:
            shape = CONVEYOR_SHAPE_2
        elif subtype == 3:
            shape = CONVEYOR_SHAPE_3
        elif subtype == 4:
            shape = CONVEYOR_SHAPE_4
        elif subtype == 5:
            shape = CONVEYOR_SHAPE_5
        elif subtype == 6:
            shape = CONVEYOR_SHAPE_6
        elif subtype == 7:
            shape = CONVEYOR_SHAPE_7

        super().__init__(position, shape)


if __name__ == "__main__":
    # filename = os.path.join(".", "tasks", "001.task.json")
    filename = os.path.join(".", "tasks", "manual solutions", "task_1.json")
    env = Environment.from_json(filename)
