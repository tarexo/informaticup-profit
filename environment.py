from shapes import *
from buildings import *

from dataclasses import dataclass
import json
import os

# create a nice output when displaying the entire grid
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


class Environment:
    def __init__(self, width, height, turns, products: dict):
        self.width = width
        self.height = height
        self.turns = turns
        self.products = products

        self.buildings = []
        self.grid = np.full((height, width), ' ')

    def add_building(self, building: Building):
        if self.is_legal_position(building):
            self.buildings.append(building)
            
            # calculate building position relative to the center of the shape
            shape_x = building.position[0] - building.shape.center[0]
            shape_y = building.position[1] - building.shape.center[1]
            
            height, width = building.shape.elements.shape
            for y_offset in range(height):
                for x_offset in range(width):
                    # free space can be ignored
                    if building.shape.elements[y_offset, x_offset] == ' ':
                        continue
                    
                    x = shape_x + x_offset
                    y = shape_y + y_offset
                    self.grid[y, x] = building.shape.elements[y_offset, x_offset]
                    
            return True
        return False

    def is_legal_position(self, building: Building):
        # calculate building position relative to the center of the shape
        shape_x = building.position[0] - building.shape.center[0]
        shape_y = building.position[1] - building.shape.center[1]
        
        height, width = building.shape.elements.shape
        for y_offset in range(height):
            for x_offset in range(width):
                # free space can be ignored
                if building.shape.elements[y_offset, x_offset] == ' ':
                    continue
                
                x = shape_x + x_offset
                y = shape_y + y_offset
                
                # check for empty positions of every individual element
                if not self.is_tile_empty(x, y):
                    return False
                
                # ToDo: check_constraints (mines cannot be placed next to other mines)
        return True
    
    def is_tile_empty(self, x, y):
        # check whether tile is out of grid bounds
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False
        
        # check for other buildings
        if self.grid[y, x] != ' ':
            return False
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
                building = Obstacle(position, obj["width"], obj["height"])
            elif obj["type"] == "deposit":
                building = Deposit(position, obj["width"], obj["height"], obj["subtype"])
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
                print(building)
                print()
            else:
                print(f"UNABLE TO PLACE {obj['type']} at position {position}\n")
        return env

    @staticmethod
    def to_json(filename):
        pass

    def __repr__(self):
        return f"\nCOMPLETE ENVIRONMENT:\n\n{self.grid}\n".replace("'", "")



if __name__ == "__main__":
    # filename = os.path.join(".", "tasks", "001.task.json")
    filename = os.path.join(".", "tasks", "manual solutions", "task_1.json")
    env = Environment.from_json(filename)
    
    print(env)
