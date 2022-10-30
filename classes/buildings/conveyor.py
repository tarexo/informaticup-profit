from .building import Building
from shapes import *

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