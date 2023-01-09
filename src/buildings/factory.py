from .building import Building
import numpy as np
import helper.simulation_logs as simlog


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

    def get_center_position(self):
        return (self.x + 2, self.y + 2)

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

    def end_of_round_action(self, product, round):
        """Executes the end of round action, produces as many products as possible with the currently available resources.

        Args:
            product (dictionary): Information about the product given by the environment class.
            round (int): Current round.

        Returns:
            int: Number of products produces.
        """
        recipe = np.array(product["resources"])
        t = self.resources - recipe

        num_products = 0
        while np.min(self.resources - recipe) >= 0:
            self.resources = self.resources - recipe
            simlog.log_factory_end_round(self, round, product["points"])
            num_products += 1

        return num_products


class SimpleFactory(Factory):
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
            "type": "simple_factory",
            "x": int(self.x),
            "y": int(self.y),
            "subtype": self.subtype,
        }
        return building_dict

    def get_center_position(self):
        return self.x, self.y

    def get_input_positions(self):
        return self.get_element_positions("F")


from model.settings import SIMPLE_GAME

if SIMPLE_GAME:
    Factory = SimpleFactory
