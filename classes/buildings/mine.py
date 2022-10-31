from .building import Building
from shapes import *

class Mine(Building):
    def __init__(self, position, subtype):
        self.subtype = subtype

        from helper.dicts.building_shapes import BUILDING_SHAPES
        shape = BUILDING_SHAPES[type(self)][subtype]
            
        super().__init__(position, shape)