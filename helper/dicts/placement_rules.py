from classes.buildings import *

LEGAL_CONNECTIONS = {
    Deposit: [Mine],
    Mine: [Conveyor, Combiner, Factory],
    Conveyor: [Conveyor, Combiner, Mine, Factory],
    Combiner: [Conveyor, Combiner, Mine, Factory],
    Factory: [],  # Factories do not have outputs
    Obstacle: [],  # Obstacles do not have outputs
}

SIMPLE_LEGAL_CONNECTIONS = {
    SimpleDeposit: [SimpleConveyor],
    SimpleConveyor: [SimpleConveyor, SimpleFactory],
    SimpleFactory: [],
}

from helper.constants.settings import SIMPLE_GAME

if SIMPLE_GAME:
    LEGAL_CONNECTIONS = SIMPLE_LEGAL_CONNECTIONS
