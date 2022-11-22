from .building import Building
import numpy as np


class Conveyor(Building):
    """The conveyor transports resources in a straight line.

    Inherits from the class Building.

    Attributes
    ----------
    position : tuple
        The position of the building in (x,y)
    shape : Shape
        The shape of the building
    resources : list
        The resources currently held by the building
    subtype : int
        The subtype of the conveyor, determining its rotation (0-7)
    """

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "conveyor",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
        }
        return building_dict

    def resource_string_builder(self, round, store_indices, cache_indices):
        str = f"{round} (start): ({self.x},{self.y}) accepts ["
        for i in cache_indices:
            str += f"{self.resource_cache[i]}x{i}, "
        str = str[:-2]
        str += "], holds ["
        for i in store_indices:
            str += f"{self.resources[i]}x{i}, "
        str = str[:-2]
        str += "]"
        return str

    def start_of_round_action(self, round):
        cache_indices = np.where(self.resource_cache > 0)[0]
        if len(cache_indices) == 0:
            return

        for i in cache_indices:
            self.resources[i] += self.resource_cache[i]
        store_indices = np.where(self.resources > 0)[0]
        print(f"{self.resource_string_builder(round, store_indices, cache_indices)}")
        self.resource_cache = np.array([0] * 8)
        return

    def end_of_round_action(self, round):
        indices = np.where(self.resources > 0)[0]
        for i in indices:
            self.connections[0].resource_cache[i] += self.resources[i]
        self.resources = np.array([0] * 8)
        return
