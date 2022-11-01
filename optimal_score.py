
from environment import Environment
import os
import json
import numpy as np

class Product:

    def __init__(self, subtype, resources, points):
        self.subtype = subtype
        self.resources = resources
        self.points = points

    @staticmethod
    def from_json(product_json):
        return Product(product_json['subtype'],product_json['resources'],product_json['points'])



def optimal_score(env):
    products = []
    product_resoucres = np.empty((len(env.products),8))
    env_resources = np.array([])
    for i in range(len(env.products)) :#get products and fill product_resouces matrix
        new_product = Product.from_json(env.products[i])
        products.append(new_product)
        product_resoucres[i] = new_product.resources
    






if __name__ == '__main__':
    filename = os.path.join(".", "tasks", "004.task.json")
    env = Environment.from_json(filename)
    optimal_score(env)