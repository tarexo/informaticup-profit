from environment.simulator import Simulator
from model.architecture import DeepQNetwork, ActorCritic
from settings import GAME_SOLVER_MODEL_NAME, DEBUG
from helper.file_handler import *
from helper.optimal_score import calculate_optimal_score
from environment.setup import set_default_options, make_gym

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
        print(f"solving task: '{filename}'...")

        optimal_score_options = calculate_optimal_score(self.env)
        optimal_score = optimal_score_options[0][0]

        for _optimal_score, sorted_products in optimal_score_options:
            for product in sorted_products:
                success = self.solve_single_product(product)
                false_targets = self.env.buildings

                self.env.make_untargetable(false_targets)

            score, turns = Simulator(self.env).run()
            if score != 0:
                print("initial solution:")
                print(self.env)
                print(f"theoretical optimal score: {int(optimal_score)}")
                print(f"our solution scored {score} points in {turns} turns")

                simple_solution = deepcopy(self.env)
                self.enhance_solution()

                new_score, new_turns = Simulator(self.env).run()
                if new_score < score or (new_score == score and new_turns > turns):
                    self.env = simple_solution
                else:
                    score = new_score
                    turns = new_turns
                break

        print("enhanced solution:")
        print(self.env)
        print("SUCCESS" if success and score != 0 else "FAILURE")
        print(f"theoretical optimal score: {int(optimal_score)}")
        print(f"our solution scored {score} points in {turns} turns")
        print("\n")

        environment_to_placeable_buildings_list(self.env, filename.split("\\")[-1])

        return success

    def solve_single_product(self, product):
        original_task = deepcopy(self.env)
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

            self.env = deepcopy(original_task)
            self.model.env = self.env
        return False

    def connect_resource_to_factory(self, deposits, factory):
        at_least_one_connection = False
        for deposit in deposits:
            self.env.remove_building(factory)
            backup_env = deepcopy(self.env)
            for mine in self.env.get_possible_mines(deposit, max=10):
                self.env.add_building(factory)
                self.env.add_building(mine)
                self.env.set_task(mine, factory)
                success = self.connect_mine_to_factory(mine, factory, backup_env)

                if success:
                    at_least_one_connection = True
                    break
        return at_least_one_connection

    def connect_mine_to_factory(self, mine, factory, backup_env):
        state = self.env.grid_to_observation()

        _, episode_reward = self.model.run_episode(
            state, exploration_rate=0, greedy=True, force_legal=True
        )

        if DEBUG:
            print(self.env)
        if episode_reward == 1:
            return True

        self.env = deepcopy(backup_env)
        self.model.env = self.env

    def enhance_solution(self):
        deposits = self.env.get_deposits()
        factories = self.env.get_factories()

        for factory in factories:
            for deposit in deposits:
                if self.env.is_connected(deposit, factory):
                    backup_env = deepcopy(self.env)
                    true_targets = [
                        b
                        for b in self.env.buildings
                        if self.env.is_connected(b, factory)
                    ]
                    for mine in self.env.get_possible_mines(deposit, max=10):
                        self.env.make_targetable(true_targets)
                        self.env.add_building(mine)
                        self.env.set_task(mine, factory)
                        success = self.connect_mine_to_factory(
                            mine, factory, backup_env
                        )

                        if success:
                            break
                    self.env.make_untargetable(true_targets)


if __name__ == "__main__":
    set_default_options()
    solver = GameSolver(model_name=GAME_SOLVER_MODEL_NAME)

    task_dir = os.path.join(".", "tasks", "cup")
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
