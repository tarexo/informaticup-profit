from shapes import *

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, order=True)   
class Building:
    position: tuple
    shape: Shape
    resources = [0]*8
    

class Obstacle(Building):
    def __init__(self, position, width, height):
        super().__init__(position, obstacle_shape(width, height))


class Deposit(Building):
    def __init__(self, position, width, height, subtype):
        super().__init__(position, deposit_shape(width, height))

        self.width = width
        self.height = height

        self.subtype = subtype
        self.resources[subtype] = self.width * self.height * 5
        
    def __repr__(self):
        #insert subtype and #resources
        s = super().__repr__()
        return s.replace(", shape=", f", subtype={self.subtype}, #resources={self.resources[self.subtype]}, shape=")
    
class Mine(Building):
    def __init__(self, position, subtype):
        self.subtype = subtype
        if subtype == 0:
            shape = MINE_SHAPE_0
        elif subtype == 1:
            shape = MINE_SHAPE_1
        elif subtype == 2:
            shape = MINE_SHAPE_2
        elif subtype == 3:
            shape = MINE_SHAPE_3
            
        super().__init__(position, shape)
        
        
class Combiner(Building):
    def __init__(self, position, subtype):
        self.subtype = subtype
        if subtype == 0:
            shape = COMBINER_SHAPE_0
        elif subtype == 1:
            shape = COMBINER_SHAPE_1
        elif subtype == 2:
            shape = COMBINER_SHAPE_2
        elif subtype == 3:
            shape = COMBINER_SHAPE_3
            
        super().__init__(position, shape)


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