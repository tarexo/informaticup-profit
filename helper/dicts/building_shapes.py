from shapes import *
from classes.buildings.mine import Mine
from classes.buildings.conveyor import Conveyor
from classes.buildings.combiner import Combiner


BUILDING_SHAPES = {
    Mine: {
        0: Shape((1, 0), np.array([[' ', 'm', 'm', ' '], ['+', 'm', 'm', '-']])),
        1: Shape((0, 1), np.array([['+', ' '], ['m', 'm'], ['m', 'm'], ['-', ' ']])),
        2: Shape((1, 0), np.array([['-', 'm', 'm', '+'], [' ', 'm', 'm', ' ']])),
        3: Shape((0, 1), np.array([[' ', '-'], ['m', 'm'], ['m', 'm'], [' ', '+']])),
    },
    Conveyor: {
        0: Shape((1, 0), np.array([['+', '>', '-']])),
        1: Shape((0, 1), np.array([['+'], ['v'], ['-']])),
        2: Shape((1, 0), np.array([['-', '<', '+']])),
        3: Shape((0, 1), np.array([['-'], ['^'], ['+']])),
        4: Shape((1, 0), np.array([['+', '>', '>', '-']])),
        5: Shape((0, 1), np.array([['+'], ['v'], ['v'], ['-']])),
        6: Shape((1, 0), np.array([['-', '<', '<', '+']])),
        7: Shape((0, 1), np.array([['-'], ['^'], ['^'], ['+']])),
    },
    Combiner: {
        0: Shape((1, 1), np.array([['+', 'c', ' '], ['+', 'c', '-'], ['+', 'c', ' ']])),
        1: Shape((1, 1), np.array([['+', '+', '+'], ['c', 'c', 'c'], [' ', '-', ' ']])),
        2: Shape((1, 1), np.array([[' ', 'c', '+'], ['-', 'c', '+'], [' ', 'c', '+']])),
        3: Shape((1, 1), np.array([[' ', '-', ' '], ['c', 'c', 'c'], ['+', '+', '+']])),
    }
}