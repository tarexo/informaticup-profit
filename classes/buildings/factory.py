from .building import Building
import numpy as np


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
        indices = np.where(self.resource_cache > 0)[0]
        for i in indices:
            self.resources[i] += self.resource_cache[i]
            print(
                f"{round} (start): ({self.x},{self.y}) accepts {self.resource_cache[i]}x{i}, holds {self.resources[i]}x{i}"
            )
            self.resource_cache[i] = 0
        return

    def end_of_round_action(self, recipe, points, round):
        recipe = np.array(recipe)
        t = self.resources - recipe

        num_products = 0
        while np.min(self.resources - recipe) >= 0:
            self.resources = self.resources - recipe
            print(
                f"{round} (end): ({self.x},{self.y}) produces {self.subtype} ({points} points)"
            )
            num_products += 1

        return num_products
