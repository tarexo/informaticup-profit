from .building import Building
from shapes import *


class Conveyor(Building):
    """The conveyor transports resources in a straight line.

    Inherits from the class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    resources : list
        The resources currently held by the building
    subtype : int
        The subtype of the conveyor, determining its rotation (0-7)
    """

    def __init__(self, position, subtype):
        """Init function to create an instance of the conveyor.

        Args:
            position (tuple): Position of the conveyor in (x,y)
            subtype (int): The subtype of the conveyor, determining its rotation (0-7)
        """
        self.subtype = subtype

        from helper.dicts.building_shapes import BUILDING_SHAPES

        shape = BUILDING_SHAPES[type(self)][subtype]

        super().__init__(position, shape)

    def to_json(self):
        building_dict = {
            "type": "conveyor",
            "x": self.position[0],
            "y": self.position[1],
            "subtype": self.subtype,
        }
        return building_dict
