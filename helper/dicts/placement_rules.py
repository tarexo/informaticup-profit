from classes.buildings import *

LEGAL_CONNECTIONS = {
    Deposit: [Mine],
    Mine: [Conveyor, Combiner, Factory],
    Conveyor: [Conveyor, Combiner, Mine, Factory],
    Combiner: [Conveyor, Combiner, Mine, Factory],
    Factory: [],  # Factories do not have outputs
    Obstacle: [],  # Obstacles do not have outputs
    SimpleDeposit: [SimpleConveyor],
    SimpleConveyor: [SimpleConveyor, SimpleFactory],
    SimpleFactory: [],
}
