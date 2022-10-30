from shapes import *
from dataclasses import dataclass

@dataclass(frozen=True, order=True)   
class Building:
    """The base class for all objects of the game (excluding the grid itself)

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    resources : list
        The resources currently held by the building
    """
    position: tuple
    shape: Shape
    resources = [0]*8