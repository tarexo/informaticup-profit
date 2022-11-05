from .building import *


class UnplacableBuilding(Building):
    """The base class for Deposits and Obstacle which are given by a specific task.
    Unlike other Buildings they comprise a width and height attribute.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    width : int
        The grid width of the building
    height : int
        The grid height of the building
    resources : list
        The resources currently held by the building
    """

    def __init__(self, x, y, subtype, width, height):
        self.width = width
        self.height = height

        from .deposit import Deposit
        from .obstacle import Obstacle

        from helper.dicts.building_shapes import (
            create_deposit_shape,
            create_obstacle_shape,
        )

        if type(self) == Deposit:
            shape = create_deposit_shape(width, height)
        elif type(self) == Obstacle:
            shape = create_obstacle_shape(width, height)

        super().__init__(x, y, subtype, shape=shape)

    @classmethod
    def from_input_position(cls, x, y, subtype):
        raise PermissionError("Unplacable buildings have no inputs!")

    def __repr__(self) -> str:
        return f"{type(self).__name__}_{self.subtype} at x={self.x}, y={self.y} with width={self.width}, height={self.height} \n{self.shape}\n"
