from .building import Building
import numpy as np
import helper.functions.simulation_logs as simlog


class Factory(Building):
    """A factory produces products using the resources it receives.

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
        The subtype of the factory, determining the product (0-7)
    """

    NUM_SUBTYPES = 8

    def to_json(self):
        building_dict = {
            "type": "factory",
            "x": self.x,
            "y": self.y,
            "subtype": self.subtype,
        }
        return building_dict

    def start_of_round_action(self, round):
        cache_indices = np.where(self.resource_cache > 0)[0]

        if len(cache_indices) == 0:
            return

        for i in cache_indices:
            self.resources[i] += self.resource_cache[i]

        store_indices = np.where(self.resources > 0)[0]
        simlog.log_start_round(self, round, store_indices, cache_indices)
        self.resource_cache = np.array([0] * 8)

    def end_of_round_action(self, recipe, points, round):
        recipe = np.array(recipe)
        t = self.resources - recipe

        num_products = 0
        while np.min(self.resources - recipe) >= 0:
            self.resources = self.resources - recipe
            simlog.log_factory_end_round(self, round, points)
            num_products += 1

        return num_products
