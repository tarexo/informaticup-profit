import unittest
import os

import helper.optimal_score as opt
import helper.file_handler as fh


class Optimal_Score_Tests(unittest.TestCase):
    def test_scores_of_environments(self):
        filename = os.path.join(".", "tasks", "cup", "001.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env), 410)

        filename = os.path.join(".", "tasks", "cup", "002.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env), 120)

        filename = os.path.join(".", "tasks", "cup", "003.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env), 60)

        filename = os.path.join(".", "tasks", "cup", "004.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env), 720)
