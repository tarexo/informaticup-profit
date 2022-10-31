from .building import Building
from shapes import *

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