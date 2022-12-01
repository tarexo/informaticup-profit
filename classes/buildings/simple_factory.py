from .building import Factory


class SimpleFactory(Factory):
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

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "simple_factory",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
        }
        return building_dict
