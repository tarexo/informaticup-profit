import classes.buildings as buildings
import os
import numpy as np
import itertools
import helper.functions.file_handler as fh


class Product:
    def __init__(self, subtype, resources, points):
        self.subtype = subtype
        self.resources = resources
        self.points = points

    @staticmethod
    def from_json(product_json):
        return Product(
            product_json["subtype"], product_json["resources"], product_json["points"]
        )


def optimal_score(env):
    products = []
    product_resoucres = np.empty((len(env.products), 8))
    env_resources = np.zeros(8)
    turns = env.turns
    for i in range(len(env.products)):  # get products and fill product_resouces matrix
        new_product = Product.from_json(env.products[i])
        products.append(new_product)
        product_resoucres[i] = new_product.resources
    for building in env.buildings:
        if building.__class__ == buildings.Deposit:
            env_resources[building.subtype] += building.width * building.height * 5
    product_combinations = []
    for i in range(len(products) + 1):  # fill combinations
        for subset in itertools.combinations(products, i):
            product_combinations.append(subset)
    best_score = 0
    for combination in product_combinations:
        temp_score = 0
        temp_env_resources = env_resources
        for i in range(turns):
            for p in combination:
                n = np.subtract(temp_env_resources, p.resources)
                if is_resource_value_negative(n):
                    break
                else:
                    temp_env_resources = n
                    temp_score += p.points
        if temp_score > best_score:
            best_score = temp_score
    return best_score


@staticmethod
def is_resource_value_negative(resources):
    for value in resources:
        if value < 0:
            return True
    return False


if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "001.task.json")
    env = fh.environment_from_json(filename)
    print(optimal_score(env))