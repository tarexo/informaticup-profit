from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, order=True)
class Shape:
    """The shape function defines the position aswell as the shape (width, height) of any building.

    Attributes
    ----------
    center : tuple
        The position of the object in (x,y) (Is not neccessarily the center of the object!)
    elements : np.ndarray
        The shape of the object, how it is presented in the grid
    """
    center: tuple
    elements: np.ndarray
    
    def __repr__(self):
        # mark center of the shape with *
        a = np.copy(self.elements)
        a[self.center[1], self.center[0]] = '*'
        return f"\n{a}"


# ' ' = free space
# '+' = input element
# '-' = output element
# 'd', 'f', 'm', 'c', 'x', '<', '>', '^', 'v' = neural element

FACTORY_SHAPE = Shape(
    (0, 0),
    np.array(
        [
            ['+', '+', '+', '+', '+'],
            ['+', 'f', 'f', 'f', '+'],
            ['+', 'f', 'f', 'f', '+'],
            ['+', 'f', 'f', 'f', '+'],
            ['+', '+', '+', '+', '+'],
        ]
    ),
)

def obstacle_shape(width, height):
    array = np.full((height, width), 'x')
    return Shape((0, 0), array)

def deposit_shape(width, height):
    array = np.full((height, width), '-')
    array[1:(height-1), 1:(width-1)] = 'd'
    return Shape((0, 0), array)
