# from environment import Environment
from classes.buildings import Factory
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

        rng = np.random.default_rng()

        buildings = self.env.buildings
        buildings_indices = np.array([i for i in range(len(buildings))])
        # Connections = buildings[i].connections
        total_points = 0
        total_rounds = 0

        for round in range(self.rounds):
            rng.shuffle(buildings_indices)

            # COMMENT: Round start
            for i in buildings_indices:
                buildings[i].start_of_round_action(round + 1)

            rng.shuffle(buildings_indices)

            # COMMENT: Round end
            for i in buildings_indices:
                if type(buildings[i]) == Factory:
                    all_products = self.env.products

                    product = None
                    for p in all_products:
                        if p["subtype"] == buildings[i].subtype:
                            product = p
                            break

                    num_products = buildings[i].end_of_round_action(
                        recipe=product["resources"],
                        points=product["points"],
                        round=round + 1,
                    )
                    if num_products * product["points"] > 0:
                        total_points += num_products * product["points"]
                        total_rounds = round + 1
                    continue

                buildings[i].end_of_round_action(round + 1)

        return (total_points, total_rounds)
