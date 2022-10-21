from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, order=True)
class Shape:
    center: tuple
    elements: np.ndarray
    
    def __repr__(self):
        a = np.where(self.elements == 0, " ", self.elements)
        a = np.where(self.elements == 1, "0", a)
        a = np.where(self.elements == 2, "+", a)
        a = np.where(self.elements == 3, "-", a)
        a[self.center[1], self.center[0]] = "*"
        return f"\n{a}"

@dataclass(frozen=True, order=True)   
class Building:
    position: tuple
    shape: Shape
    resources = [0]*8

# 0 = free space
# 1 = neural element
# 2 = input element
# 3 = output element

FACTORY_SHAPE = Shape(
    (0, 0),
    np.array(
        [
            [3, 3, 3, 3, 3],
            [3, 1, 1, 1, 3],
            [3, 1, 1, 1, 3],
            [3, 1, 1, 1, 3],
            [3, 1, 1, 1, 3],
            [3, 3, 3, 3, 3],
        ]
    ),
)

CONVEYOR_SHAPE_0 = Shape((1, 0), np.array([[3, 1, 2]]))
CONVEYOR_SHAPE_1 = Shape((0, 1), np.array([[3], [1], [2]]))
CONVEYOR_SHAPE_2 = Shape((1, 0), np.array([[2, 1, 3]]))
CONVEYOR_SHAPE_3 = Shape((0, 1), np.array([[2], [1], [3]]))
CONVEYOR_SHAPE_4 = Shape((1, 0), np.array([[3, 1, 1, 2]]))
CONVEYOR_SHAPE_5 = Shape((0, 1), np.array([[3], [1], [1], [2]]))
CONVEYOR_SHAPE_6 = Shape((1, 0), np.array([[2, 1, 1, 3]]))
CONVEYOR_SHAPE_7 = Shape((0, 1), np.array([[2], [1], [1], [3]]))

# ToDo: shape should be adjustable for obstacle/deposit
# Placeholder:
OBSTACLE_SHAPE = Shape((0, 0), np.array([[1]]))
DEPOSIT_SHAPE = Shape((0, 0), np.array([[2, 2, 2], [2, 1, 2], [2, 2, 2]]))

