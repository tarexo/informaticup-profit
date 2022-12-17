import environment
import os
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh
from  optimal_score import *
import collections
import random

class GameSolver:
    def __init__(self,env:environment.Environment):
        self.env = env
        self.orderd_products = self.get_product_order()
        self.optimal_score = optimal_score(env)
        print(self.optimal_score)
        #self.solve()



    def update(self, pairs):
        print('to be done')
        #update product order if one or more pairs don't work

    def get_product_order(self):
        order = []
        products = get_products(self.env)
        all_resources = get_env_resources(self.env)
        turns = self.env.turns
        for p in products:
            score = calc_best_score([p], turns, all_resources)
            order.append([p,score])
        order = self.sort_product_list(order)
        return order


    def sort_product_list(self,order):
        sorted = order.copy()
        nums = np.zeros(len(order))
        for i in range(len(order)):
            nums[i] = order[i][1]
            
        args = np.argsort(nums)
        for i in range(len(args)):
            n = len(args)-1-i
            sorted[i] = order[args[n]]
        return sorted

    def solve(self):
        print('solve Game')
        #build pairs
        for pair in self.pairs:
            success = self.build_pairs(pair)
            if success == False:
                self.update(pair)
            else: self.pairs.remove(pair)
        



    def build_pair(self):
        n = random.randint(0,1)
        if n == 1: return True
        return False

if __name__ == "__main__":
    filename = os.path.join(".", "tasks","hard", "profit.task.1671272940276.json")#tasks\hard\profit.task.1671032210813.json
    env = fh.environment_from_json(filename)
    solver = GameSolver(env)
    