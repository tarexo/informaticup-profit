from classes.buildings import *

POSITIONAL_ACTION_TO_DIRECTION = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}

BUILDING_ACTION_TO_CLASS_SUBTYPE = {
    0: (Mine, 0),
    1: (Mine, 1),
    2: (Mine, 2),
    3: (Mine, 3),
    4: (Combiner, 0),
    5: (Combiner, 1),
    6: (Combiner, 2),
    7: (Combiner, 3),
    8: (Conveyor, 0),
    9: (Conveyor, 1),
    10: (Conveyor, 2),
    11: (Conveyor, 3),
    12: (Conveyor, 4),
    13: (Conveyor, 5),
    14: (Conveyor, 6),
    15: (Conveyor, 7),
    16: (Conveyor, 8),
}
