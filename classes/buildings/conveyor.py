from .building import Building
import numpy as np
import helper.functions.simulation_logs as simlog


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

    def start_of_round_action(self, round):
        """Executes the start of round action, adding all resources from the cache to the resources array.

        Args:
            round (int): Current round.
        """
        cache_indices = np.where(self.resource_cache > 0)[0]

        if len(cache_indices) == 0:
            return

        for i in cache_indices:
            self.resources[i] += self.resource_cache[i]

        store_indices = np.where(self.resources > 0)[0]
        simlog.log_start_round(self, round, store_indices, cache_indices)
        self.resource_cache = np.array([0] * 8)

    def end_of_round_action(self, round):
        """Executes the end of round action pushing all resources to the cache of the next building.

        Args:
            round (int): Current round.
        """
        indices = np.where(self.resources > 0)[0]
        for i in indices:
            self.connections[0].resource_cache[i] += self.resources[i]
        self.resources = np.array([0] * 8)
