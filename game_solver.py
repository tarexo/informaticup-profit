from optimal_score import *
from helper.functions.file_handler import *

from model import *
from profit_gym import make_gym, register_gym

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

    def solve_task(self, task_name):
        filename = os.path.join(".", "tasks", task_name)
        self.env.from_json(filename)

        # ToDo: sort_products
        # for product in self.env.get_products():
        product = {}
        product["subtype"] = 0
        product["resources"] = [1, 1, 1, 0, 0, 0, 0, 0]
        success = self.solve_single_product(filename, product)
        print(success)
        print(self.env)

    def solve_single_product(self, filename, product):
        original_task = deepcopy(self.env)
        for factory in self.env.get_possible_factories(product["subtype"]):
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

    def connect_resource_to_factory(self, deposits, factory):
        resource_task = deepcopy(self.env)
        for deposit in deposits:
            for mine in self.env.get_possible_mines(deposit):
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
    solver = GameSolver(model_name="NORMAL__12x12__DQN_512-3x3_128")
    solver.solve_task("test_task.json")
