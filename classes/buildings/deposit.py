from .unplacable_building import UnplacableBuilding
import numpy as np


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

    def __init__(self, position, subtype, width, height):
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
        return

    def end_of_round_action(self, round):
        if np.max(self.resources) == 0 or len(self.connections) == 0:
            return

        indices = np.array([i for i in range(len(self.connections))])
        for i in indices:
            self.connections[i].resource_cache[self.subtype] += (
                3 if self.resources[self.subtype] >= 3 else self.resources[self.subtype]
            )
            took = (
                3 if self.resources[self.subtype] >= 3 else self.resources[self.subtype]
            )
            print(
                f"{round} (end): ({self.x},{self.y}) takes {took}x{i}, {self.resources[self.subtype] - took}x{i} available"
            )
            self.resources[self.subtype] -= 3
            if self.resources[self.subtype] <= 0:
                break
        return
