import unittest
import os
import optimal_score as opt
import helper.functions.file_handler as fh
import simulator
import environment


class Simulator_Tests(unittest.TestCase):
    def test_simulator_task_1(self):
        filename1 = os.path.join(".", "tasks", "manual solutions", "task_1.json")
        env = fh.environment_from_json(filename1)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 410)
        self.assertEqual(total_rounds, 43)

    def test_simulator_task_1(self):
        filename2 = os.path.join(".", "tasks", "manual solutions", "task_2.json")
        env = fh.environment_from_json(filename2)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 90)
        self.assertEqual(total_rounds, 20)

    def test_simulator_task_2(self):
        filename3 = os.path.join(".", "tasks", "manual solutions", "task_3.json")
        env = fh.environment_from_json(filename3)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 40)
        self.assertEqual(total_rounds, 14)

    def test_simulator_task_3_non_opt(self):
        filename3_non_opt = os.path.join(
            ".", "tasks", "manual solutions", "task_3_non_optimal.json"
        )
        env = fh.environment_from_json(filename3_non_opt)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 20)
        self.assertEqual(total_rounds, 13)

    def test_simulator_task_4(self):
        filename4 = os.path.join(
            ".", "tasks", "manual solutions", "task_4.json"
        )  # tasks\manual solutions\task_4.json
        env = fh.environment_from_json(filename4)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 120)
        self.assertEqual(total_rounds, 47)


if __name__ == "__main__":
    unittest.main()
