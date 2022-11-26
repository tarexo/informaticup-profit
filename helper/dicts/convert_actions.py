from classes.buildings import *

POSITIONAL_ACTION_TO_DIRECTION = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}

BUILDING_ACTION_TO_CLASS_SUBTYPE = {
    0: (Conveyor, 0),
    1: (Conveyor, 1),
    2: (Conveyor, 2),
    3: (Conveyor, 3),
    4: (Conveyor, 4),
    5: (Conveyor, 5),
    6: (Conveyor, 6),
    7: (Conveyor, 7),
    8: (Combiner, 0),
    9: (Combiner, 1),
    10: (Combiner, 2),
    11: (Combiner, 3),
    12: (Mine, 0),
    13: (Mine, 1),
    14: (Mine, 2),
    15: (Mine, 3),
}
