from shapes import *
from .building import Building

class Obstacle(Building):
    """An obstacle is an inanimate object on the grid placed by the game, not by the player. Other buildings cannot be placed on it.

    Inherits from class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    """
    def __init__(self, position, width, height):
        super().__init__(position, obstacle_shape(width, height))