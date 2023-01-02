import environment
import os
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh
from  optimal_score import *
import collections
import random
import copy
from helper.functions.building_placer import *

class GameSolver:
    def __init__(self,env:environment.Environment):
        self.env = env
        #self.orderd_products = self.get_product_order()
        self.score_list= optimal_score(env)
        self.env_resources = get_env_resources(env)
        self.all_deposits= get_deposits(env)
    

        
    def solve(self):
        print('solve Game')
        #build pairs
        product_list = self.score_list[0]
        products_scores = []
        for p in product_list[1]:
            score = calc_best_score_of_single_product(env, p, self.env.turns -2, self.env_resources)
            products_scores.append([score,p])
        products_scores = self.sort_product_list(products_scores)

        for _,p in products_scores:
            deposits = []
            
            index = np.argwhere(p.resources>0)
            for d in self.all_deposits:
                if d.subtype in index:
                    deposits.append(d)
            connected = self.solve_product(p,deposits)
            if not connected : break 
            
    
    def solve_product(self, product:Product, deposits):
        factory_positions = get_all_factory_positions(self.env,self.all_deposits)
        connected = False
        tmp_env = copy.copy(self.env)
        for factory_pos in factory_positions:
            for deposit in deposits:
                factory = buildings.Mine((factory_pos[0], factory_pos[1]), product.subtype)
                success = self.env.add_building(factory)
                if success == None: continue
                factory_positions.remove(factory_pos)
                connected = self.make_connection(deposit, factory)
                if not connected:
                    self.env = tmp_env
                    break
        if not connected :
            self.update(product)
            return False
        return True
                  

    def make_connection(self, deposit, factory):
        mine_positions = get_all_mines_positions(self.env, deposit)
        for mine_pos in mine_positions:
            #build mine
            mine = buildings.Mine((mine_pos[0], mine_pos[1]), mine_pos[2])
            success = self.env.add_building(mine)
            if success == None:continue
            connected = self.build_connection(mine, factory)
            if connected == True: return True
        return False

    def update(self, product):
        new_score_list = []
        for products in self.score_list:
            if product not in products:
                new_score_list.append(products)
        self.score_list = new_score_list
        self.solve

        #update product order if one or more pairs don't work



    def sort_product_list(self,order):
        sorted = order.copy()
        nums = np.zeros(len(order))
        for i in range(len(order)):
            nums[i] = order[i][0]
            
        args = np.argsort(nums)
        for i in range(len(args)):
            n = len(args)-1-i
            sorted[i] = order[args[n]]
        return sorted

    def build_connection(self, mine, factory):
        print("Mine: "+str(mine.x)+", "+str(mine.y)+" Subtype: "+str(mine.subtype) +"  to Factory: "+str(factory.x)+", "+str(factory.y)+" Subtype: "+str(factory.subtype) )
        return True
        '''n = random.randint(0,1)
        if n == 1: return True
        return False'''

if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "easy","profit.task.1672670811891.json")#tasks\easy\profit.task.1672670811891.json
    env = fh.environment_from_json(filename)
    solver = GameSolver(env)
    solver.solve()

    #deposits = get_deposits(env)
    #for deposit in deposits:
    #    mine_positions = get_all_mines_positions(env, deposit)
    #    for x,y,sub in mine_positions:
    #       print(str(x)+', '+str(y)+':  '+str(sub))
    #factory_positions = get_all_factory_positions(env,deposits)
    #for a in factory_positions:
        #print(a)
    
    