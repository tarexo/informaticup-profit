import classes.buildings as buildings
import os
import numpy as np
import itertools
import math
import helper.functions.file_handler as fh


class Product:
    def __init__(self, subtype, resources, points):
        self.subtype = subtype
        self.resources = resources
        self.points = points

    @staticmethod
    def from_json(product_json):
        subtype = product_json["subtype"]
        resouces = np.array(product_json["resources"])
        points = product_json["points"]
        return Product(
           subtype ,resouces , points
        )


def optimal_score(env):
    products = get_products(env)
    env_resources = get_env_resources(env)
    product_scores =[]
    turns = env.turns #get resource to mine and from mine to factory takes 2 turn minimum
    product_combinations = []
    for i in range(len(products) + 1):  # fill combinations
        for subset in itertools.combinations(products, i):
            product_combinations.append(subset)
    for combination in product_combinations:
        score = calc_best_score(env,combination, turns, env_resources)
        product_scores = add_to_score_list(product_scores,score, combination)
    return product_scores

def add_to_score_list(product_scores,score, combination):
    if len(product_scores) ==0:
        product_scores.append([score, combination])
        return product_scores
    insert_at = len(product_scores)
    for i in range(len(product_scores)):
        if product_scores[i][0]< score:
            insert_at = i
            break
    product_scores.insert(insert_at,[score, combination])
    return product_scores


def calc_best_score(env,combination, turns, all_resources):
    myturns = turns-2
    resources = all_resources.copy()
    if len(combination)== 0: return 0
    if len(combination)== 1: return calc_best_score_of_single_product(env,combination[0], myturns, resources)
    indices = []
    quotient = np.zeros(len(resources))
    for product in combination:
        v=np.argwhere(product.resources>0)
        for a in v:
            indices.append(a)
    for i in indices:
        quotient[i]+=1
    quotient=np.where(quotient>1,quotient,1)
    new_resouces = resources/quotient
    score = 0
    for p in combination:
        score+=calc_best_score_of_single_product(env,p, myturns, new_resouces)
    return score
    
    
    

def calc_best_score_of_single_product(env, product:Product, myturns, resources):
    score = 0
    mines = np.array([0,0,0,0,0,0,0,0])
    ind = np.argwhere(product.resources>0)
    points = product.points
    deposits = get_deposits(env)
    turns = np.zeros(product.resources.shape)
    product_resouces = product.resources

    for d in deposits:
        for i in ind:
            if(d.subtype)== i:
                mines[i] += d.width + d.height
    for i in range(len(mines)):
        if mines[i]== 0:
            turns[i]=0
        else:
            turns[i]=np.ceil(resources[i]/(mines[i]*3))
    products_build = []
    if np.amax(turns)<= myturns:
        for i in range(len(resources)):
            if product_resouces[i]>0:
                products_build.append(np.floor(resources[i]/product_resouces[i]))
        products_build = np.array(products_build)
        score = np.amin(products_build)*points
    else:
        a=myturns*mines*3
        for i in range(len(resources)):
            if product_resouces[i]>0:
                products_build.append(np.floor(a[i]/product_resouces[i]))
        products_build = np.array(products_build)
        score = np.amin(products_build)*points
    return score



def get_products(env):
    products = []
    for i in range(len(env.products)):  # get products and fill product_resouces matrix
        new_product = Product.from_json(env.products[i])
        products.append(new_product)
    return products

def get_deposits(env):
    deposits = []
    for building in env.buildings:
        if building.__class__ == buildings.Deposit:
            deposits.append(building)
    return deposits

def get_env_resources(env):
    env_resources = np.zeros(8)
    deposits =get_deposits(env)
    for d in deposits:
        env_resources[d.subtype] += d.width * d.height * 5
    return env_resources

def is_resource_value_negative(resources):
    for value in resources:
        if value < 0:
            return True
    return False

def set_procduct_resouces(a):
    r = np.zeros(a.shape)
    index = np.argwhere(a>0)
    for i in index:
        r[i] = 1
    return r

if __name__ == "__main__":
    filename = os.path.join(".", "tasks", "004.task.json")
    env = fh.environment_from_json(filename)
    score_list =  optimal_score(env)
    for s in score_list:
        print(s[0])
        for p in s[1]:
            print(p.subtype)
        print('\n')

