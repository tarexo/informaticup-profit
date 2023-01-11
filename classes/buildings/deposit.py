from .unplacable_building import UnplacableBuilding
import numpy as np
import helper.functions.simulation_logs as simlog


class Deposit(UnplacableBuilding):
    """A deposit holds one of eight possible resources.

    Inherits from class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    width : int
        The width of the deposit
    height : int
        The height of the deposit
    resources : list
        The resources currently held by the building. Initialized with width * height * 5
    subtype : int
        The subtype of the deposit, determining its held resource (0-7)
    """

    def __init__(self, position, subtype, width=3, height=3):
        super().__init__(position, subtype, width, height)
        self.resources[subtype] += 5 * width * height

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "deposit",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
            "width": self.width,
            "height": self.height,
        }
        return building_dict

    def start_of_round_action(self, round):
        """Empty function for cleaner code in simulator.py.

        Args:
            round (int): Current round.
        """
        return

    def end_of_round_action(self, round):
        """Executes end of round action, pushing 3 items of the resource given by self.subtype to all connected mines.

        Args:
            round (int): Current round.
        """
        if np.max(self.resources) == 0 or len(self.connections) == 0:
            return

        indices = np.array([i for i in range(len(self.connections))])
        for i in indices:
            self.connections[i].resource_cache[self.subtype] += (
                3 if self.resources[self.subtype] >= 3 else self.resources[self.subtype]
            )
            takes_out = (
                3 if self.resources[self.subtype] >= 3 else self.resources[self.subtype]
            )
            simlog.log_deposit_end_round(self, round, takes_out)
            self.resources[self.subtype] -= 3
            if self.resources[self.subtype] <= 0:
                break
        return


class SimpleDeposit(Deposit):
    """A deposit holds one of eight possible resources.

    Inherits from class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    width : int
        The width of the deposit
    height : int
        The height of the deposit
    resources : list
        The resources currently held by the building. Initialized with width * height * 5
    subtype : int
        The subtype of the deposit, determining its held resource (0-7)
    """

    def __init__(self, position, subtype, width=1, height=1):
        super().__init__(position, subtype, 1, 1)
        from shapes import Shape

        self.shape = Shape(0, 0, np.array([["D"]]))
        self.resources[subtype] += 5

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "simple_deposit",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
            "width": self.width,
            "height": self.height,
        }
        return building_dict

    def get_output_positions(self):
        return self.get_element_positions("D")


from helper.constants.settings import SIMPLE_GAME

if SIMPLE_GAME:
    Deposit = SimpleDeposit
