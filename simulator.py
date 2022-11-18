from environment import Environment


class Simulator:
    """
    Simulates the game.
    """

    def __init__(self, env: Environment, rounds: int = 0, time: int = 0):
        """Initialize the simulator

        Args:
            env (Environment): Environment on which to run the simulation
            rounds (int, optional): Maximum number of rounds to simulate (0 = no restriction). Defaults to 0.
            time (int, optional): Maximum time to run the simulation in seconds (0 = no restriction). Defaults to 0.
        """
        self.env = env
        self.rounds = rounds
        self.time = time

    def run(self):
        # NOTE: This implementation will for now always run the number of rounds given by rounds (will not abort if no more points are possible)
        # TODO: Start game loop

        # TODO: Execute round start actions for each building
        # TODO: Execute round end actions for each building
        # TODO: Check if points were generated and add them to overall score

        return  # TODO: Return overall score
