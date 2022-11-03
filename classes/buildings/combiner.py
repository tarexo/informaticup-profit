from .building import Building
from shapes import *


class Combiner(Building):
    """The combiner has multiple inputs and combines them in one output.

    Inherits from class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    resources : list
        The resources currently held by the building
    subtype : int
        The subtype of the combiner, determining its rotation (0-3)
    """

    NUM_SUBTYPES = 4

    def __init__(self, position, subtype):
        """Init function to create an instance of the combiner.

        Args:
            position (tuple): Position of the combiner in (x,y)
            subtype (int): The subtype of the combiner, determining its rotation (0-3)
        """
        self.subtype = subtype

        from helper.dicts.building_shapes import BUILDING_SHAPES

        shape = BUILDING_SHAPES[type(self)][subtype]

        super().__init__(position, shape)