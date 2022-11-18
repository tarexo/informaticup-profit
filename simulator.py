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
        # TODO: Check if rounds is set and limit game loop accordingly

        # TODO: Check if time is set and limit game loop accordingly

        # TODO: Start game loop

        # TODO: Execute round start actions for each building
        # TODO: Execute round end actions for each building
        # TODO: Check if points were generated and add them to overall score

        return  # TODO: Return overall score
