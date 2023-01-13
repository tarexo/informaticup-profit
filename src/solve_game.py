from environment.simulator import Simulator
from model.architecture import DeepQNetwork, ActorCritic
from settings import GAME_SOLVER_MODEL_NAME, DEBUG
from helper.file_handler import *
from helper.optimal_score import calculate_optimal_score
from environment.setup import set_default_options, make_gym

from copy import copy, deepcopy
import sys
import time


class GameSolver:
    def __init__(self, model_name):
        model_path = os.path.join(".", "saved_models", model_name)
        _game_type, field_of_vision, network = model_name.split("__")
        field_of_vision = int(field_of_vision.split("x")[0])

        self.env = make_gym(20, 20, field_of_vision)
        self.env.reset()

        if "DQN" in network:
            self.model = DeepQNetwork(self.env)
        elif "A-C" in network:
            self.model = ActorCritic(self.env)
        self.model.load(model_path)

    def solve_task(self, filename, reset_clock=False):
        if reset_clock:
            global START_TIME
            START_TIME = time.time()

        self.env.from_json(filename)
        self.model.env = self.env
        self.original_task = deepcopy(self.env)

        print(self.env)
        print(f"solving task: '{filename}'...")

        optimal_score_options = calculate_optimal_score(self.env)
        optimal_score, sorted_products = optimal_score_options[0]
        self.optimal_score = int(optimal_score)

        score, turns = self.create_initial_solution(sorted_products)
        self.evaluate_solution("initial solution", score, turns)

        score, turns = self.create_enhanced_solution(score, turns)

        environment_to_placeable_buildings_list(self.env, os.path.split(filename)[1])
        environment_to_json(self.env, os.path.split(filename)[1])

        solve_time = time.time() - START_TIME
        if solve_time > self.env.time:
            print("WARNING: time limit has been exceeded!")
        print(
            f"It took {round(solve_time, 2)}s of the allowed {self.env.time}s to calculate this solution.\n\n"
        )

        return score > 0

    def create_initial_solution(self, sorted_products):
        at_least_one_product = False
        for product in sorted_products:
            success = self.solve_single_product(product)
            if success:
                at_least_one_product = True

        self.env.make_targetable(self.env.buildings)

        score, turns = Simulator(self.env).run()
        if not at_least_one_product:
            assert score == 0

        return score, turns

    def create_enhanced_solution(self, initial_score, initial_turns):
        best_solution = simple_solution = deepcopy(self.env)
        best_score = initial_score
        best_turns = initial_turns

        for iteration in range(10):
            if (time.time() - START_TIME) / self.env.time > 0.5:
                print("no further enhancements due to time constraint!")
                break

            success = self.enhance_solution()
            self.env.make_targetable(self.env.buildings)

            if not success:
                print("no further enhancements could be made\n")
                return best_score, best_turns

            score, turns = Simulator(self.env).run()
            self.evaluate_solution(f"enhanced solution #{iteration+1}", score, turns)

            if score < best_score or (score == best_score and turns >= best_turns):
                print(
                    "reverting back to previous solution due to a lack of improvement\n"
                )
                self.env = best_solution
                return best_score, best_turns
            else:
                best_solution = deepcopy(self.env)
                best_score = score
                best_turns = turns

        return best_score, best_turns

    def solve_single_product(self, product):
        backup_buildings = copy(self.env.buildings) + copy(self.env.obstacles)
        for factory in self.env.get_possible_factories(product.subtype, max=15):
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
            backup_buildings = copy(self.env.buildings) + copy(self.env.obstacles)
            for mine in self.env.get_possible_mines(deposit, factory, max=20):
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

        return False

    def enhance_solution(self):
        deposits = self.env.get_deposits()
        factories = self.env.get_factories()

        any_enhancement = False

        for factory in factories:
            for deposit in deposits:
                if self.env.is_connected(deposit, factory):
                    backup_buildings = copy(self.env.buildings) + copy(
                        self.env.obstacles
                    )
                    for mine in self.env.get_possible_mines(deposit, factory, max=10):
                        self.env.add_building(mine)
                        self.env.set_task(mine, factory)
                        if self.env.is_connected(mine, factory):
                            success = True
                            any_enhancement = True
                            break

                        self.determine_targets(factory)
                        success = self.connect_mine_to_factory()

                        if success:
                            any_enhancement = True
                            break

                        self.restore_backup(backup_buildings)

        return any_enhancement

    def evaluate_solution(self, name, score, turns):
        print(f"{name}:")
        print(self.env)
        print(f"theoretical optimal score: {self.optimal_score}")
        print(f"our solution scored {score} points in {turns} turns\n")

    def determine_targets(self, factory):
        for building in self.env.buildings:
            if building == factory or self.env.is_connected(building, factory):
                self.env.make_targetable([building])
            else:
                self.env.make_untargetable([building])

    def restore_backup(self, backup_buildings):
        self.env = deepcopy(self.original_task)
        self.env.empty()
        self.model.env = self.env

        for building in backup_buildings:
            building.clear_connections()
        self.env.add_buildings(backup_buildings)


def solve_test_tasks(directory, sleep_in_between_tasks=8):
    task_dir = os.path.join(".", "tasks", directory)
    tasks = [
        f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))
    ]

    solved_tasks = 0
    for task_name in tasks:
        filename = os.path.join(task_dir, task_name)
        success = solver.solve_task(filename, reset_clock=True)
        if success:
            solved_tasks += 1
        time.sleep(sleep_in_between_tasks)

    print(f"successfully solved tasks: {solved_tasks}/{len(tasks)}")


if __name__ == "__main__":
    startup_time = 3  # time for importing all necessary libraries
    START_TIME = time.time() - startup_time

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
        for directory in ["cup", "easy", "hard"]:
            solve_test_tasks(directory)
