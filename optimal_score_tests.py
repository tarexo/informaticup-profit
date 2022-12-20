import unittest
import os
from environment import Environment
import optimal_score as opt
import helper.functions.file_handler as fh


class Optimal_Score_Tests(unittest.TestCase):
    def test_scores_of_environments(self):
        filename = os.path.join(".", "tasks", "001.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env)[0][0], 410)

        filename = os.path.join(".", "tasks", "002.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env)[0][0], 120)

        filename = os.path.join(".", "tasks", "003.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env)[0][0], 60)

        filename = os.path.join(".", "tasks", "004.task.json")
        env = fh.environment_from_json(filename)
        self.assertEqual(opt.optimal_score(env)[0][0], 720)


if __name__ == "__main__":
    unittest.main()
