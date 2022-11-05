from .building import Building


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
