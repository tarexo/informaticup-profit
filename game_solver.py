import environment
import os
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh
from  optimal_score import *
import collections
import random
from helper.functions.building_placer import *

class GameSolver:
    def __init__(self,env:environment.Environment):
        self.env = env
        #self.orderd_products = self.get_product_order()
        self.score_list= optimal_score(env)
        print(self.optimal_score)
        #self.solve()

        
    def solve(self):
        print('solve Game')
        #build pairs
        for product_list in self.score_list:
            products_scores = []
            for p in product_list[1]:
                score = calc_best_score_of_single_product(env, p, self.env.turns -2, self.env.resources)
                products_scores.append([score,p])
            products_scores = self.sort_product_list(products_scores)

            for p in products_scores:
                deposits = []
                all_deposits= get_deposits(env)
                for d in all_deposits:
                    if d.subtype == p.subtype:
                        deposits.append(d)
                self.solve_product(p,deposits)
            
    
    def solve_product(self, product:Product, deposits):
        min_connections = np.count_nonzero(product.resources)
        #build factory
        factory_positions = get_all_factory_positions(self.env)
        #build mine
        for deposit in deposits:
            mine_positions = get_all_mines_positions(self.env, deposit)
            #build connection

                



        '''for pair in self.pairs:
            success = self.build_pairs(pair)
            if success == False:
                self.update(pair)
            else: self.pairs.remove(pair)'''


    def update(self, pairs):
        print('to be done')
        #update product order if one or more pairs don't work

    '''def get_product_order(self):
        order = []
        products = get_products(self.env)
        all_resources = get_env_resources(self.env)
        turns = self.env.turns
        for p in products:
            score = calc_best_score([p], turns, all_resources)
            order.append([p,score])
        order = self.sort_product_list(order)
        return order'''


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

    def build_pair(self):
        n = random.randint(0,1)
        if n == 1: return True
        return False

if __name__ == "__main__":
    filename = os.path.join(".", "tasks","004.task.json")#tasks\hard\profit.task.1671032210813.json
    env = fh.environment_from_json(filename)
    deposits = get_deposits(env)
    for deposit in deposits:
        mine_positions = get_all_mines_positions(env, deposit)
    #factory_positions = get_all_factory_positions(env,deposits)
    #for a in factory_positions:
        #print(a)
    
    