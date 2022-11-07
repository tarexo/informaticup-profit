from .unplacable_building import UnplacableBuilding


class Obstacle(UnplacableBuilding):
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
