from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, order=True)
class Shape:
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

CONVEYOR_SHAPE_0 = Shape((1, 0), np.array([['+', '>', '-']]))
CONVEYOR_SHAPE_1 = Shape((0, 1), np.array([['+'], ['v'], ['-']]))
CONVEYOR_SHAPE_2 = Shape((1, 0), np.array([['-', '<', '+']]))
CONVEYOR_SHAPE_3 = Shape((0, 1), np.array([['-'], ['^'], ['+']]))
CONVEYOR_SHAPE_4 = Shape((1, 0), np.array([['+', '>', '>', '-']]))
CONVEYOR_SHAPE_5 = Shape((0, 1), np.array([['+'], ['v'], ['v'], ['-']]))
CONVEYOR_SHAPE_6 = Shape((1, 0), np.array([['-', '<', '<', '+']]))
CONVEYOR_SHAPE_7 = Shape((0, 1), np.array([['-'], ['^'], ['^'], ['+']]))


MINE_SHAPE_0 = Shape((1, 0), np.array([[' ', 'm', 'm', ' '], ['+', 'm', 'm', '-']]))
MINE_SHAPE_1 = Shape((0, 1), np.array([['+', ' '], ['m', 'm'], ['m', 'm'], ['-', ' ']]))
MINE_SHAPE_2 = Shape((1, 0), np.array([['-', 'm', 'm', '+'], [' ', 'm', 'm', ' ']]))
MINE_SHAPE_3 = Shape((0, 1), np.array([[' ', '-'], ['m', 'm'], ['m', 'm'], [' ', '+']]))

COMBINER_SHAPE_0 = Shape((1, 1), np.array([['+', 'c', ' '], ['+', 'c', '-'], ['+', 'c', ' ']]))
COMBINER_SHAPE_1 = Shape((1, 1), np.array([['+', '+', '+'], ['c', 'c', 'c'], [' ', '-', ' ']]))
COMBINER_SHAPE_2 = Shape((1, 1), np.array([[' ', 'c', '+'], ['-', 'c', '+'], [' ', 'c', '+']]))
COMBINER_SHAPE_3 = Shape((1, 1), np.array([[' ', '-', ' '], ['c', 'c', 'c'], ['+', '+', '+']]))

def obstacle_shape(width, height):
    array = np.full((height, width), 'x')
    return Shape((0, 0), array)

def deposit_shape(width, height):
    array = np.full((height, width), '-')
    array[1:(height-1), 1:(width-1)] = 'd'
    return Shape((0, 0), array)
