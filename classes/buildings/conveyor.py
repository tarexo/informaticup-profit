from .building import Building


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

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "conveyor",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
        }
        return building_dict
