from .unplacable_building import Deposit


class SimpleDeposit(Deposit):
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

    def __init__(self, position, subtype):
        super().__init__(position, subtype, 1, 1)
        self.resources[subtype] += 5

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "simple_deposit",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
            "width": self.width,
            "height": self.height,
        }
        return building_dict
