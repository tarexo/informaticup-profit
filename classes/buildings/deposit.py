from .building import Building
from shapes import *

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
   