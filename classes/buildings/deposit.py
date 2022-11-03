from .building import Building
from shapes import *


class Deposit(Building):
    """A deposit holds one of eight possible resources.

    Inherits from class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    width : int
        The width of the deposit
    height : int
        The height of the deposit
    resources : list
        The resources currently held by the building. Initialized with width * height * 5
    subtype : int
        The subtype of the deposit, determining its held resource (0-7)
    """

    NUM_SUBTYPES = 8

    def __init__(self, position, width, height, subtype):
        """Init function to create an instance of the deposit.

        Args:
            position (tuple): Position of the deposit in (x,y)
            width (int): Width of the deposit
            height (int): Height of the deposit
            subtype (int): The subtype of the deposit, determining its held resource (0-7)
        """
        from helper.dicts.building_shapes import deposit_shape

        super().__init__(position, deposit_shape(width, height))

        self.width = width
        self.height = height

        self.subtype = subtype
        self.resources[subtype] = self.width * self.height * 5
