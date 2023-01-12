from environment.simulator import Simulator
from model.architecture import DeepQNetwork, ActorCritic
from settings import GAME_SOLVER_MODEL_NAME, DEBUG
from helper.file_handler import *
from helper.optimal_score import calculate_optimal_score
from environment.setup import set_default_options, make_gym

from copy import copy, deepcopy
import sys


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
        self.original_task = deepcopy(self.env)

        print(self.env)
        print(f"solving task: '{filename}'...")

        optimal_score_options = calculate_optimal_score(self.env)
        self.optimal_score = int(optimal_score_options[0][0])

        for _, sorted_products in optimal_score_options:
            at_least_one_product = False
            for product in sorted_products:
                success = self.solve_single_product(product)
                if success:
                    at_least_one_product = True

            score, turns = Simulator(self.env).run()
            if not at_least_one_product:
                assert score == 0
                continue

            self.evaluate_solution("initial solution", score, turns)

            simple_solution = deepcopy(self.env)
            for i in range(3):
                self.enhance_solution()

            self.evaluate_solution("enhanced solution", score, turns)

            new_score, new_turns = Simulator(self.env).run()
            if new_score < score or (new_score == score and new_turns >= turns):
                print("reverting back to initial solution due to no improvments")
                # self.env = simple_solution
            else:
                score = new_score
                turns = new_turns
            break

        environment_to_placeable_buildings_list(self.env, filename.split("\\")[-1])
        environment_to_json(self.env, filename.split("\\")[-1])

        return success and score > 0

    def solve_single_product(self, product):
        backup_buildings = copy(self.env.buildings)
        for factory in self.env.get_possible_factories(product.subtype, max=10):
            self.env.add_building(factory)
            for deposit_subtype, amount in enumerate(product.resources):
                if amount == 0:
                    continue
                deposits = self.env.get_deposits(deposit_subtype)
                success = self.connect_resource_to_factory(deposits, factory)

                if not success:
                    break

            if success:
                return True

            self.restore_backup(backup_buildings)
        return False

    def connect_resource_to_factory(self, deposits, factory):
        at_least_one_connection = False
        for deposit in deposits:
            backup_buildings = copy(self.env.buildings)
            for mine in self.env.get_possible_mines(deposit, max=10):
                self.env.add_building(mine)
                if self.env.is_connected(mine, factory):
                    success = True
                    at_least_one_connection = True
                    break

                self.determine_targets(factory)
                self.env.set_task(mine, factory)
                success = self.connect_mine_to_factory()

                if success:
                    at_least_one_connection = True
                    break

                self.restore_backup(backup_buildings)

        return at_least_one_connection

    def connect_mine_to_factory(self):
        state = self.env.grid_to_observation()

        _, episode_reward = self.model.run_episode(
            state, exploration_rate=0, greedy=True, force_legal=True
        )

        if episode_reward == 1:
            return True

    def enhance_solution(self):
        deposits = self.env.get_deposits()
        factories = self.env.get_factories()

        for factory in factories:
            for deposit in deposits:
                if self.env.is_connected(deposit, factory):
                    backup_buildings = copy(self.env.buildings)
                    for mine in self.env.get_possible_mines(deposit, max=10):
                        self.env.add_building(mine)
                        self.env.set_task(mine, factory)
                        if self.env.is_connected(mine, factory):
                            success = True
                            break
                        self.determine_targets(factory)
                        success = self.connect_mine_to_factory()

                        if success:
                            break

                        self.restore_backup(backup_buildings)

    def evaluate_solution(self, name, score, turns):
        print(f"{name}:")
        print(self.env)
        print(f"theoretical optimal score: {self.optimal_score}")
        print(f"our solution scored {score} points in {turns} turns")
        print("\n")

    def determine_targets(self, factory):
        for building in self.env.buildings:
            if building == factory or self.env.is_connected(building, factory):
                self.env.make_targetable([building])
            else:
                self.env.make_untargetable([building])

    def restore_backup(self, backup_buildings):
        self.env = deepcopy(self.original_task)
        self.model.env = self.env
        self.env.add_buildings(backup_buildings)


def solve_test_tasks():
    task_dir = os.path.join(".", "tasks", "wrong_score_turns")
    tasks = [
        f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))
    ]

    solved_tasks = 0
    for task_name in tasks:
        filename = os.path.join(task_dir, task_name)
        success = solver.solve_task(filename)
        if success:
            solved_tasks += 1

    print(f"successfully solved tasks: {solved_tasks}/{len(tasks)}")


if __name__ == "__main__":
    set_default_options()
    solver = GameSolver(model_name=GAME_SOLVER_MODEL_NAME)

    if len(sys.argv) == 2:
        filename = sys.argv[0]
        if not os.path.is_file(filename):
            print(f"'{filename}' is no correct filename")
            print("Usage: python solve_game.py [filename]")
            print("if no filename is provided, all test tasks will be solved instead")
        solver.solve_task(sys.argv[0])
    else:
        solve_test_tasks()
