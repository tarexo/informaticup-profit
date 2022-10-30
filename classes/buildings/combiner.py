from .building import Building
from shapes import *

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