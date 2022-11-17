import unittest
import os
from environment import Environment
import classes.buildings as buildings
import numpy as np
import helper.functions.file_handler as fh

filename3 = os.path.join(".", "tasks", "003.task.json")
filename2 = os.path.join(".", "tasks", "002.task.json")


class Environment_Tests(unittest.TestCase):
    def test_initialize_environment(self):

        env = fh.environment_from_json(filename2)
        with open("env_test_task002.npy", "rb") as f:
            expected = np.load(f)
        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

        env = fh.environment_from_json(filename3)

        with open("env_test_task003.npy", "rb") as f:
            expected = np.load(f)
        np.testing.assert_array_equal(env.grid, expected)
        self.assertEqual(env.grid.shape, expected.shape)

    def test_placing_factory_empty_field(self):
        # place building at empty field
        env2 = fh.environment_from_json(filename2)
        factory = buildings.Factory((0, 0), 1)
        self.assertEqual(
            env2.add_building(factory),
            None,
            "A factory should not be allowed at the same position as a deposit",
        )

    def test_placing_two_factories_same_position(self):
        env3 = fh.environment_from_json(filename3)
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

    def test_mine_subtypes(self):
        env3 = fh.environment_from_json(filename3)
        mine = buildings.Mine((8, 2), 0)
        self.assertEqual(
            env3.add_building(mine),
            None,
            "mines are not allowed to overlap with deposits",
        )
        env3 = fh.environment_from_json(filename3)
        mine = buildings.Mine((8, 2), 1)
        self.assertEqual(
            env3.add_building(mine),
            mine,
            "mine should be allowed to be build next to deposits",
        )

    def test_mine_next_to_conveyor(self):
        env3 = fh.environment_from_json(filename3)
        mine = buildings.Mine((9, 4), 0)
        self.assertEqual(env3.add_building(mine), mine)
        combiner = buildings.Combiner((9, 7), 0)
        self.assertEqual(env3.add_building(combiner), None)

    def test_mine_next_to_mine(self):
        env3 = fh.environment_from_json(filename3)
        mine = buildings.Mine((9, 4), 0)
        self.assertEqual(env3.add_building(mine), mine)
        mine2 = buildings.Mine((13, 4), 0)
        self.assertEqual(env3.add_building(mine2), None)

    def test_combiner_next_to_deposit(self):
        env3 = fh.environment_from_json(filename3)
        combiner = buildings.Combiner((9, 5), 3)
        self.assertEqual(
            env3.add_building(combiner),
            None,
            "combiners are not allowed next to deposits",
        )

    def test_conveyor_next_to_deposit(self):
        env3 = fh.environment_from_json(filename3)
        conveyor = buildings.Combiner((9, 1), 0)
        self.assertEqual(
            env3.add_building(conveyor),
            None,
            "conveyors are not allowed next to deposits",
        )

    def test_factory_next_to_deposit(self):
        env3 = fh.environment_from_json(filename3)
        factory = buildings.Combiner((8, 2), 0)
        self.assertEqual(
            env3.add_building(factory),
            None,
            "factories are not allowed next to deposits",
        )

    def test_factory_next_to_factory(self):
        env3 = fh.environment_from_json(filename3)
        factory1 = buildings.Combiner((12, 1), 0)
        self.assertEqual(env3.add_building(factory1), factory1)
        factory2 = buildings.Combiner((17, 1), 0)
        self.assertEqual(
            env3.add_building(factory2),
            factory2,
            "factories should be allowed to be build next to a facotry",
        )

    def test_deposit_mine_conveyor_conveyor(self):
        env3 = fh.environment_from_json(filename3)
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

    def test_deposit_mine_conveyor_conveyor_illegal_conveyor1(self):
        env3 = fh.environment_from_json(filename3)
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
        conv3 = buildings.Conveyor((7, 12), 4)
        self.assertEqual(
            env3.add_building(conv3),
            None,
            "only one conveyor is allowed at a mine exit",
        )

    def test_deposit_mine_conveyor_conveyor_illegal_conveyor2(self):
        env3 = fh.environment_from_json(filename3)
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
        conv3 = buildings.Conveyor((10, 10), 0)
        self.assertEqual(
            env3.add_building(conv3),
            None,
            "only one conveyor is allowed at a conveyor exit",
        )

    def test_combiner_next_to_combiner(self):
        env3 = fh.environment_from_json(filename3)
        combiner1 = buildings.Combiner((14, 6), 0)
        self.assertEqual(
            env3.add_building(combiner1),
            combiner1,
            "combiners are allowed to be placed in an empty space",
        )
        combiner2 = buildings.Combiner((17, 6), 0)
        self.assertEqual(
            env3.add_building(combiner2),
            combiner2,
            "combiners are allowed to be connected with a combiner exit",
        )


if __name__ == "__main__":
    unittest.main()
