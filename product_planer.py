import environment
import os
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh
from  optimal_score import *
import collections

class ProductPlaner:
    def __init__(self,env:environment.Environment):
        self.env = env


    def update(self, pairs):
        print('to be done')
        #update product order if one or more pairs don't work

    def get_product_order(self):
        order = []
        products = get_products(self.env)
        all_resources = get_env_resources(self.env)
        turns = self.env.turns
        print(turns)
        for p in products:
            score = calc_best_score([p], turns, all_resources)
            order.append([p,score])
        order = self.sort_product_list(order)


    def sort_product_list(self,order):
        sorted = []
        nums = np.zeros(len(order))
        for i in range(len(order)):
            nums[i] = order[i][1]
        args = np.argsort(nums)
        print(args)
        



if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "004.task.json")
    env = fh.environment_from_json(filename)
    product_planer = ProductPlaner(env)
    product_planer.get_product_order()