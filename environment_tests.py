import unittest
import os
from environment import Environment
import buildings
import numpy as np


class Environment_Tests(unittest.TestCase):
    def test_initialize_environment(self):
        filename = os.path.join(".", "tasks", "002.task.json")
        env = Environment.from_json(filename)
        with open("env_test_task002.npy", "rb") as f:
            expected = np.load(f)
        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

        filename = os.path.join(".", "tasks", "003.task.json")
        env = Environment.from_json(filename)

        with open("env_test_task003.npy", "rb") as f:
            expected = np.load(f)
        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

    def test_placing_buildings(self):
        filename = os.path.join(".", "tasks", "002.task.json")
        env2 = Environment.from_json(filename)
        filename = os.path.join(".", "tasks", "003.task.json")
        env3 = Environment.from_json(filename)
        factory = buildings.Factory((0, 0), 1)
        self.assertFalse(env2.add_building(factory))

        factory = buildings.Factory((10, 10), 1)
        self.assertTrue(env3.add_building(factory))
        self.assertFalse(env3.add_building(factory))

        # mine subtye
        env3 = Environment.from_json(filename)
        mine = buildings.Mine((8, 2), 0)
        self.assertFalse(
            env3.add_building(mine), "mines are not allowed to overlap with deposits"
        )
        env3 = Environment.from_json(filename)
        mine = buildings.Mine((8, 2), 1)
        self.assertTrue(
            env3.add_building(mine),
            "mine should be allowed to be build next to deposits",
        )

    def test_building_constrains(self):
        filename = os.path.join(".", "tasks", "003.task.json")
        # mine next to combiner
        env3 = Environment.from_json(filename)
        mine = buildings.Mine((9, 4), 0)
        self.assertTrue(env3.add_building(mine))
        combiner = buildings.Combiner((9, 7), 0)
        self.assertFalse(env3.add_building(combiner))

        # mine next to mine
        env3 = Environment.from_json(filename)
        self.assertTrue(env3.add_building(mine))
        mine2 = buildings.Mine((13, 4), 0)
        self.assertFalse(env3.add_building(mine2))

        # combiner to deposit
        env3 = Environment.from_json(filename)
        combiner = buildings.Combiner((9, 5), 3)
        self.assertFalse(
            env3.add_building(combiner), "combiners are not allowed next to deposits"
        )

        # conveyor to deposit
        env3 = Environment.from_json(filename)
        conveyor = buildings.Combiner((9, 1), 0)
        self.assertFalse(
            env3.add_building(conveyor), "conveyors are not allowed next to deposits"
        )

        # factory to deposit
        env3 = Environment.from_json(filename)
        factory = buildings.Combiner((8, 2), 0)
        self.assertFalse(
            env3.add_building(factory), "factories are not allowed next to deposits"
        )

        # factory to factory
        env3 = Environment.from_json(filename)
        factory1 = buildings.Combiner((12, 1), 0)
        self.assertTrue(env3.add_building(factory1))
        factory2 = buildings.Combiner((17, 1), 0)
        self.assertTrue(
            env3.add_building(factory2),
            "factories should be allowed to be build next to a facotry",
        )

        # deposit->mine->conveyor->conveyor
        env3 = Environment.from_json(filename)
        mine = buildings.Mine((8, 5), 3)
        self.assertTrue(
            env3.add_building(mine),
            "mines should be allowed to be build next to a deposits",
        )
        conv1 = buildings.Conveyor((10, 7), 0)
        self.assertTrue(
            env3.add_building(conv1),
            "conveyors should be allowed to be build next to a mines",
        )
        conv2 = buildings.Conveyor((13, 7), 0)
        self.assertTrue(
            env3.add_building(conv2),
            "conveyors should be allowed to be build next to a conveyors",
        )


if __name__ == "__main__":
    unittest.main()
