import unittest
import os
import helper.file_handler as fh
import environment.simulator as simulator

filename1 = os.path.join(".", "tasks", "cup", "manual solutions", "task_1.json")
filename2 = os.path.join(".", "tasks", "cup", "manual solutions", "task_2.json")
filename3 = os.path.join(".", "tasks", "cup", "manual solutions", "task_3.json")
filename4 = os.path.join(".", "tasks", "cup", "manual solutions", "task_4.json")


class Simulator_Tests(unittest.TestCase):
    def test_simulator_task_1(self):
        env = fh.environment_from_json(filename1)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 410)
        self.assertEqual(total_rounds, 43)

    def test_simulator_task_1(self):
        env = fh.environment_from_json(filename2)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 90)
        self.assertEqual(total_rounds, 20)

    def test_simulator_task_2(self):
        env = fh.environment_from_json(filename3)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 40)
        self.assertEqual(total_rounds, 14)

    def test_simulator_task_4(self):
        env = fh.environment_from_json(filename4)
        sim = simulator.Simulator(env)
        total_points, total_rounds = sim.run()
        self.assertEqual(total_points, 120)
        self.assertEqual(total_rounds, 47)
