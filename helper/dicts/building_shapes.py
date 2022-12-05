from shapes import *
from classes.buildings import *


# ' ' = free space
# '+' = input element
# '-' = output element
# 'd', 'f', 'm', 'c', 'x', '<', '>', '^', 'v' = neural element

BUILDING_SHAPES = {
    Mine: {
        0: Shape(1, 0, np.array([[" ", "m", "m", " "], ["+", "m", "m", "-"]])),
        1: Shape(0, 1, np.array([["+", " "], ["m", "m"], ["m", "m"], ["-", " "]])),
        2: Shape(1, 0, np.array([["-", "m", "m", "+"], [" ", "m", "m", " "]])),
        3: Shape(0, 1, np.array([[" ", "-"], ["m", "m"], ["m", "m"], [" ", "+"]])),
    },
    Conveyor: {
        0: Shape(1, 0, np.array([["+", ">", "-"]])),
        1: Shape(0, 1, np.array([["+"], ["v"], ["-"]])),
        2: Shape(1, 0, np.array([["-", "<", "+"]])),
        3: Shape(0, 1, np.array([["-"], ["^"], ["+"]])),
        4: Shape(1, 0, np.array([["+", ">", ">", "-"]])),
        5: Shape(0, 1, np.array([["+"], ["v"], ["v"], ["-"]])),
        6: Shape(1, 0, np.array([["-", "<", "<", "+"]])),
        7: Shape(0, 1, np.array([["-"], ["^"], ["^"], ["+"]])),
    },
    SimpleConveyor: {
        0: Shape(0, 0, np.array([["+", "-"]])),
        1: Shape(0, 0, np.array([["+"], ["-"]])),
        2: Shape(0, 0, np.array([["-", "+"]])),
        3: Shape(0, 0, np.array([["-"], ["+"]])),
        4: Shape(1, 0, np.array([["+", ">", "-"]])),
        5: Shape(0, 1, np.array([["+"], ["v"], ["-"]])),
        6: Shape(1, 0, np.array([["-", "<", "+"]])),
        7: Shape(0, 1, np.array([["-"], ["^"], ["+"]])),
    },
    Combiner: {
        0: Shape(1, 1, np.array([["+", "c", " "], ["+", "c", "-"], ["+", "c", " "]])),
        1: Shape(1, 1, np.array([["+", "+", "+"], ["c", "c", "c"], [" ", "-", " "]])),
        2: Shape(1, 1, np.array([[" ", "c", "+"], ["-", "c", "+"], [" ", "c", "+"]])),
        3: Shape(1, 1, np.array([[" ", "-", " "], ["c", "c", "c"], ["+", "+", "+"]])),
    },
    Factory: dict(
        enumerate(
            [
                Shape(
                    0,
                    0,
                    np.array(
                        [
                            ["+", "+", "+", "+", "+"],
                            ["+", "f", "f", "f", "+"],
                            ["+", "f", "f", "f", "+"],
                            ["+", "f", "f", "f", "+"],
                            ["+", "+", "+", "+", "+"],
                        ]
                    ),
                )
            ]
            * 8
        )
    ),
    SimpleFactory: dict(
        enumerate(
            [
                Shape(
                    0,
                    0,
                    np.array([["+"]]),
                )
            ]
            * 8
        )
    ),
}


def create_obstacle_shape(width, height):
    """Generate the shape of the obstacle class.

    Args:
        width (int): Width of the obstacle
        height (int): Height of the obstacle

    Returns:
        np.ndarray: The shape of the obstacle, how it will be presented on the grid.
    """
    array = np.full((height, width), "x")
    return Shape(0, 0, array)


def create_deposit_shape(width, height):
    """Generate the shape of the deposit class.

    Args:
        width (int): Width of the deposit
        height (int): Height of the deposit

    Returns:
        np.ndarray: The shape of the deposit, how it will be presented on the grid.
    """
    array = np.full((height, width), "-")
    array[1 : (height - 1), 1 : (width - 1)] = "d"
    return Shape(0, 0, array)
