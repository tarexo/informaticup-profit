from .building import Building
import numpy as np


class Combiner(Building):
    """The combiner has multiple inputs and combines them in one output.

    Inherits from class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    resources : list
        The resources currently held by the building
    subtype : int
        The subtype of the combiner, determining its rotation (0-3)
    """

    NUM_SUBTYPES = 4

    def to_json(self):
        building_dict = {
            "type": "combiner",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
        }
        return building_dict

    def start_of_round_action(self, round):
        indices = np.where(self.resource_cache > 0)[0]
        for i in indices:
            self.resources[i] += self.resource_cache[i]
            print(
                f"{round} (start): ({self.x},{self.y}) accepts {self.resource_cache[i]}x{i}, holds {self.resources[i]}x{i}"
            )
            self.resource_cache[i] = 0
        return

    def end_of_round_action(self, round):
        indices = np.where(self.resources > 0)[0]
        for i in indices:
            self.connections[0].resource_cache[i] += self.resources[i]
        self.resources = np.array([0] * 8)
        return
