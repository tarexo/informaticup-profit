import unittest
import os
from optimal_score import *
import helper.functions.file_handler as fh
import environment
from helper.functions.building_placer import *
import numpy as np


class GameSolver_Tests(unittest.TestCase):

    def test_mine_positions(self):
        filename = os.path.join(".", "tasks", "easy","profit.task.1672670811891.json")
        env = fh.environment_from_json(filename)
        deposits = get_deposits(env)
        self.assertEqual(len(deposits),1)
        mine_positions = env.get_possible_mines(deposits[0], max=30)
        self.assertEqual(len(mine_positions),28)
        for mine in mine_positions:
            tuple = [mine.x,mine.y, mine.subtype]
            self.assertTrue(tuple in mine_positions_expected)
        

        


mine_positions_expected =[
        [19,5,0],
        [20,5,0],
        [21,5,0],
        [22,6,0],
        [22,7,0],
        [22,8,0],
        [21,9,0],

        [21,8,1],
        [21,9,1],
        [21,10,1],
        [18,11,1], 
        [19,11,1],
        [20,11,1],
        [17,10,1],

        [16,10,2],
        [17,10,2],
        [18,10,2],
        [15,7,2],
        [15,8,2],
        [15,9,2],
        [16,6,2],

        [16,5,3],
        [16,6,3],
        [16,7,3],
        [17,4,3],
        [18,4,3],
        [19,4,3],
        [20,5,3]
    ]













if __name__ == "__main__":
    unittest.main()