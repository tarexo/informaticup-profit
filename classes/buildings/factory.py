from .building import Building
from shapes import *

class Factory(Building):
    def __init__(self, position, subtype):
        super().__init__(position, FACTORY_SHAPE)

        self.subtype = subtype

        # product_recipe, points --> should be managed by environment or factory class?!