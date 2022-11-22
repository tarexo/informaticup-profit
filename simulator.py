# from environment import Environment
from classes.buildings import Factory, Obstacle
import numpy as np


class Simulator:
    """
    Simulates the game.
    """

    def __init__(self, env):
        """Initialize the simulator

        Args:
            env (Environment): Environment on which to run the simulation
            rounds (int, optional): Maximum number of rounds to simulate (0 = no restriction). Defaults to 0.
            time (int, optional): Maximum time to run the simulation in seconds (0 = no restriction). Defaults to 0.
        """
        self.env = env
        self.rounds = env.turns

    def run(self):
        # NOTE: This implementation will for now always run the number of rounds given by <rounds> (will not abort if no more points are possible)
        total_points = 0
        total_rounds = 0

        rng = np.random.default_rng()

        buildings = self.env.buildings
        buildings_indices = np.array([i for i in range(len(buildings))])

        products_dict = {}
        for i in range(8):
            for product in self.env.products:
                if product["subtype"] == i:
                    products_dict[i] = product

        for round in range(1, self.rounds + 1):
            rng.shuffle(buildings_indices)

            # COMMENT: Round start
            for i in buildings_indices:
                if type(buildings[i]) == Obstacle:
                    continue

                buildings[i].start_of_round_action(round)

            rng.shuffle(buildings_indices)

            # COMMENT: Round end
            for i in buildings_indices:
                if type(buildings[i]) == Obstacle:
                    continue

                if type(buildings[i]) != Factory:
                    buildings[i].end_of_round_action(round)
                    continue

                product = products_dict[buildings[i].subtype]
                num_products = buildings[i].end_of_round_action(product, round)

                if num_products * product["points"] > 0:
                    total_points += num_products * product["points"]
                    total_rounds = round

        return (total_points, total_rounds)
