from classes.buildings import *
from helper.constants.settings import SIMPLE_GAME, NUM_DIRECTIONS, NUM_SUBBUILDINGS
from pandas import DataFrame

POSITIONAL_ACTION_TO_DIRECTION = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}

POSITIONAL_ACTION_TO_COMPASS = {
    0: ">",
    1: "v",
    2: "<",
    3: "^",
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

BUILDING_ACTION_TO_ID = {
    0: "Conv >",
    1: "Conv v",
    2: "Conv <",
    3: "Conv ^",
    4: "Conv >>",
    5: "Conv vv",
    6: "Conv <<",
    7: "Conv ^^",
    8: "Comb >",
    9: "Comb v",
    10: "Comb <",
    11: "Comb ^",
    12: "Mine >",
    13: "Mine v",
    14: "Mine <",
    15: "Mine ^",
}

SIMPLE_BUILDING_ACTION_TO_CLASS_SUBTYPE = {
    0: (SimpleConveyor, 0),
    1: (SimpleConveyor, 1),
    2: (SimpleConveyor, 2),
    3: (SimpleConveyor, 3),
}

SIMPLE_BUILDING_ACTION_TO_ID = {
    0: "Conv >",
    1: "Conv v",
    2: "Conv <",
    3: "Conv ^",
}


if SIMPLE_GAME:
    BUILDING_ACTION_TO_CLASS_SUBTYPE = SIMPLE_BUILDING_ACTION_TO_CLASS_SUBTYPE
    BUILDING_ACTION_TO_ID = SIMPLE_BUILDING_ACTION_TO_ID


INDICES = POSITIONAL_ACTION_TO_COMPASS.values()
COLUMNS = BUILDING_ACTION_TO_ID.values()


def named_array(array):
    reshaped_q_values = array.numpy().reshape((NUM_DIRECTIONS, NUM_SUBBUILDINGS))
    return DataFrame(reshaped_q_values, index=INDICES, columns=COLUMNS).round(3)


def action_to_description(direction_id, subbuilding_id):
    direction = POSITIONAL_ACTION_TO_COMPASS[direction_id]
    subbuilding = BUILDING_ACTION_TO_ID[subbuilding_id]
    return str((direction, subbuilding))
