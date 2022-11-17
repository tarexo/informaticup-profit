import unittest
import os
from environment import Environment
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh


class Environment_Tests(unittest.TestCase):
    def test_initialize_environment(self):
        filename = os.path.join(".", "tasks", "002.task.json")
        env = fh.environment_from_json(filename)
        with open("env_test_task002.npy", "rb") as f:
            expected = np.load(f)
        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

        filename = os.path.join(".", "tasks", "003.task.json")
        env = fh.environment_from_json(filename)

        with open("env_test_task003.npy", "rb") as f:
            expected = np.load(f)
        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

    def test_placing_buildings(self):
        # place building at empty field
        filename = os.path.join(".", "tasks", "002.task.json")
        env2 = fh.environment_from_json(filename)
        factory = buildings.Factory((0, 0), 1)
        self.assertEqual(
            env2.add_building(factory),
            None,
            "A factory should not be allowed at the same position as a deposit",
        )

        # place twice a factory
        filename = os.path.join(".", "tasks", "003.task.json")
        env3 = fh.environment_from_json(filename)
        factory1 = buildings.Factory((10, 10), 1)
        self.assertEqual(
            env3.add_building(factory1),
            factory1,
            "A factory sould be allowd to be placed on an empty field",
        )
        factory2 = buildings.Factory((10, 10), 1)
        self.assertEqual(
            env3.add_building(factory2),
            None,
            "A factory can not be placed twice at the same position",
        )

        # mine subtye
        env3 = fh.environment_from_json(filename)
        mine = buildings.Mine((8, 2), 0)
        self.assertEqual(
            env3.add_building(mine),
            None,
            "mines are not allowed to overlap with deposits",
        )
        env3 = fh.environment_from_json(filename)
        mine = buildings.Mine((8, 2), 1)
        self.assertEqual(
            env3.add_building(mine),
            mine,
            "mine should be allowed to be build next to deposits",
        )

    def test_building_constrains(self):
        filename = os.path.join(".", "tasks", "003.task.json")
        # mine next to combiner
        env3 = fh.environment_from_json(filename)
        mine = buildings.Mine((9, 4), 0)
        self.assertEqual(env3.add_building(mine), mine)
        combiner = buildings.Combiner((9, 7), 0)
        self.assertEqual(env3.add_building(combiner), None)

        # mine next to mine
        env3 = fh.environment_from_json(filename)
        self.assertEqual(env3.add_building(mine), mine)
        mine2 = buildings.Mine((13, 4), 0)
        self.assertEqual(env3.add_building(mine2), None)

        # combiner to deposit
        env3 = fh.environment_from_json(filename)
        combiner = buildings.Combiner((9, 5), 3)
        self.assertEqual(
            env3.add_building(combiner),
            None,
            "combiners are not allowed next to deposits",
        )

        # conveyor to deposit
        env3 = fh.environment_from_json(filename)
        conveyor = buildings.Combiner((9, 1), 0)
        self.assertEqual(
            env3.add_building(conveyor),
            None,
            "conveyors are not allowed next to deposits",
        )

        # factory to deposit
        env3 = fh.environment_from_json(filename)
        factory = buildings.Combiner((8, 2), 0)
        self.assertEqual(
            env3.add_building(factory),
            None,
            "factories are not allowed next to deposits",
        )

        # factory to factory
        env3 = fh.environment_from_json(filename)
        factory1 = buildings.Combiner((12, 1), 0)
        self.assertEqual(env3.add_building(factory1), factory1)
        factory2 = buildings.Combiner((17, 1), 0)
        self.assertEqual(
            env3.add_building(factory2),
            factory2,
            "factories should be allowed to be build next to a facotry",
        )

        # deposit->mine->conveyor->conveyor
        env3 = fh.environment_from_json(filename)
        mine = buildings.Mine((6, 9), 1)
        self.assertEqual(
            env3.add_building(mine),
            mine,
            "mines should be allowed to be build next to a deposits",
        )
        conv1 = buildings.Conveyor((8, 11), 0)
        self.assertEqual(
            env3.add_building(conv1),
            conv1,
            "conveyors should be allowed to be build next to a mines",
        )
        conv2 = buildings.Conveyor((11, 11), 4)
        self.assertEqual(
            env3.add_building(conv2),
            conv2,
            "conveyors should be allowed to be build next to a conveyors",
        )


if __name__ == "__main__":
    unittest.main()
