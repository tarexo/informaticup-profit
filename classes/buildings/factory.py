from .building import Building
from shapes import *


class Factory(Building):
    """A factory produces products using the resources it receives.

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
        The subtype of the factory, determining the product (0-7)
    """

    def __init__(self, position, subtype):
        """Init function to create an instance of the factory.

        Args:
            position (tuple): Position of the factory in (x,y)
            subtype (int): The subtype of the factory, determining the product (0-7)
        """
        super().__init__(position, FACTORY_SHAPE)

        self.subtype = subtype

        # product_recipe, points --> should be managed by environment or factory class?!

    def to_json(self):
        building_dict = {
            "type": "factory",
            "x": self.position[0],
            "y": self.position[1],
            "subtype": self.subtype,
        }
        return building_dict
