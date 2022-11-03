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

    NUM_SUBTYPES = 1

    def __init__(self, position, width, height):
        """Init function to create an instance of the obstacle.

        Args:
            position (tuple): Position of the obstacle in (x,y)
            width (int): The width of the obstacle
            height (int): The height of the obstacle
        """
        from helper.dicts.building_shapes import obstacle_shape

        super().__init__(position, obstacle_shape(width, height))
