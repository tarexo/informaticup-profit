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
    turns = env.turns #get resource to mine and from mine to factory takes 2 turn minimum
    product_combinations = []
    for i in range(len(products) + 1):  # fill combinations
        for subset in itertools.combinations(products, i):
            product_combinations.append(subset)
    best_score = 0
    for combination in product_combinations:
        temp_score = calc_best_score(env,combination, turns, env_resources)
        print('\n\n')
        if temp_score > best_score:
            best_score = temp_score
    return best_score

def calc_best_score(env,combination, turns, all_resources):
    myturns = turns-2
    resources = all_resources.copy()
    if len(combination)==0:
        return 0
    elif len(combination)==1:
        return calc_best_score_of_single_product(env, combination[0], myturns, resources)
    
    ind = np.argwhere(combination[0].resources>0)
    for i in range(1,len(combination)):
        np.append(ind,np.argwhere(combination[i].resources>0))
    l=len(ind)
    ind = np.unique(ind)
    if l == len(ind):
        score = 0
        for p in combination:
            score+=calc_best_score_of_single_product(env,p, myturns, resources)
        return score
    
    
    

def calc_best_score_of_single_product(env, product:Product, myturns, resources):
    score = 0
    mines = np.array([0,0,0,0,0,0,0,0])
    ind = np.argwhere(product.resources>0)
    points = product.points
    product_resouces= np.array([0,0,0,0,0,0,0,0])
    deposits = get_deposits(env)
    turns = np.zeros(product.resources.shape)
    
    product_resouces += product.resources
    for d in deposits:
        for i in ind:
            if(d.subtype)== i:
                mines[i] += d.width + d.height
    for i in range(len(turns)):
        m = mines[i]
        if m == 0:
            turns[i]=0
        else:
            turns[i]= np.ceil(resources[i]/m*3)
    products_build = []
    if np.amax(turns)<= myturns:
        for i in range(len(turns)):
            pr = product_resouces[i]
            if pr == 0:
                turns[i]=0
            else:
                products_build.append(np.floor(resources[i]/pr))

        products_build = np.array(products_build)
        score = np.amin(products_build)*points
    else:
        a=myturns*mines*3
        for i in range(len(turns)):
            pr = product_resouces[i]
            if pr == 0:
                turns[i]=0
            else:
                products_build.append(np.floor(a/product.resources))
        products_build = np.array(products_build)
        score = np.amin(products_build)*points
    return score
    '''counter = []
    rest =[]
    for _ in range(len(combination)):
        counter.append(0)
        rest.append(0)
    while(myturns!=0):
        for i in range(len(combination)):
            product = combination[i]
            highest_value =np.amax(product.resources)
            product_resource = set_procduct_resouces(product.resources)
            val = 3
            tmp = np.subtract(resources,product_resource*val)
            if np.amin(tmp)<0:
                val = 2
                tmp = np.subtract(resources,product_resource*val)
                if np.amin(tmp)<0:
                    val = 1
                    tmp = np.subtract(resources,product_resource)
                    if np.amin(tmp)<0:
                        continue
            rest[i] += val
            if  rest[i]>=highest_value:
                n = rest[i]//highest_value
                counter[i] +=n
                rest[i]= rest[i]%highest_value      
            resources = tmp  
        myturns-=1
        if resources.max()==0:break 
    return score'''


    '''temp_score= 0
    temp_env_resources = all_resources
    for _ in range(turns):
            for p in combination:
                n = #subtract_resources(temp_env_resources, p.resources)
                if is_resource_value_negative(n):
                    break
                else:
                    temp_env_resources = n
                    temp_score += p.points
    return temp_score'''

def calc_best_score_of_multible_products(env, products, myturns, resources):
    score = 0
    mines = np.array([0,0,0,0,0,0,0,0])


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
    filename = os.path.join(".", "tasks","hard", "profit.task.1671364203528.json")#tasks\easy\profit.task.1671031243173.json
    env = fh.environment_from_json(filename)
    print(optimal_score(env))

