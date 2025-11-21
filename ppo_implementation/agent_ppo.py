from car_racing_env import CarRacingV3Wrapper
import torch
import numpy as np

class Agent_PPO():
    def __init__(self, env: CarRacingV3Wrapper, args):
        """
        TODO:
        """
        # Store the passed env as member variables
        self.env: CarRacingV3Wrapper = env

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # TODO: Init member variables from args

        # TODO: Init Deep Learning Model

        # TODO: Init Loss Function, Optimizer


    def train(self):
        """
        TODO:
        """
        # Reset the env and get the first state
        state = self.env.reset()

        self.env.step(np.array([0.0, 2.0, 0.0]))