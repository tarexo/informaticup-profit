from helper.file_handler import *
from model.model import DeepQNetwork, ActorCritic
from environment.profit_gym import make_gym, register_gym

import numpy as np
import tensorflow as tf
from copy import deepcopy


class GameSolver:
    def __init__(self, model_name):
        model_path = os.path.join(".", "saved_models", model_name)
        game_type, field_of_vision, network = model_name.split("__")
        field_of_vision = int(field_of_vision.split("x")[0])

        self.env = make_gym(20, 20, field_of_vision)
        self.env.reset()

        if "DQN" in network:
            self.model = DeepQNetwork(self.env)
        elif "A-C" in network:
            self.model = ActorCritic(self.env)
        self.model.load(model_path)

    def solve_task(self, filename):
        self.env.from_json(filename)

        print(self.env)
        print("solving task...")

        # ToDo: sort_products
        for product in self.env.products:
            success = self.solve_single_product(product)
            false_targets = self.env.buildings

            self.env.make_untargetable(false_targets)

        print(self.env)
        print("SUCCESS" if success else "FAILURE")
        print("\n")

        environment_to_placeable_buildings_list(self.env, filename.split("\\")[-1])

        return success

    def solve_single_product(self, product):
        original_task = deepcopy(self.env)
        for factory in self.env.get_possible_factories(product["subtype"], max=10):
            self.env.add_building(factory)
            for deposit_subtype, amount in enumerate(product["resources"]):
                if amount == 0:
                    continue
                deposits = self.env.get_deposits(deposit_subtype)
                success = self.connect_resource_to_factory(deposits, factory)

                if not success:
                    break

            if success:
                return True

            self.env = deepcopy(original_task)
            self.model.env = self.env
        return False

    def connect_resource_to_factory(self, deposits, factory):
        self.env.remove_building(factory)
        resource_task = deepcopy(self.env)
        for deposit in deposits:
            for mine in self.env.get_possible_mines(deposit, max=30):
                self.env.add_building(factory)
                self.env.add_building(mine)
                self.env.set_task(mine, factory)

                state = self.env.grid_to_observation()

                _, episode_reward = self.model.run_episode(
                    state, exploration_rate=0, greedy=True, force_legal=True
                )

                print(self.env)
                if episode_reward == 1:
                    return True

                self.env = deepcopy(resource_task)
                self.model.env = self.env
        return False


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    np.random.seed(42)
    tf.random.set_seed(42)
    # for some reason eager execution is not enabled in my (Leo's) installation
    tf.config.run_functions_eagerly(True)
    # suppress AVX_WARNING!
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    register_gym()
    solver = GameSolver(model_name="NORMAL__15x15__DQN_256_128")

    task_dir = os.path.join(".", "tasks", "easy")
    tasks = [
        f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))
    ]

    score = 0
    for task_name in tasks:
        filename = os.path.join(task_dir, task_name)
        success = solver.solve_task(filename)
        if success:
            score += 1

    print(f"Score: {score}/{len(tasks)}")
