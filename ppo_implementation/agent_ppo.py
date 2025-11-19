from car_racing_env import CarRacingV3Wrapper

class Agent_PPO():
    def __init__(self, env: CarRacingV3Wrapper, args):
        """
        TODO:
        """
        # Store the passed env as member variables
        self.env: CarRacingV3Wrapper = env

        # TODO: Init member variables from args

        # TODO: Init Deep Learning Model

        # TODO: Init Loss Function, Optimizer


    def train(self):
        """
        TODO:
        """
        # Reset the env
        state = self.env.reset()