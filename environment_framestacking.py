import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation
import numpy as np


class CarRacingV3Wrapper(gym.Env):
    """
    Customizable Environment for the CarRacing-v3 env from Gymnasium.
    """

    def __init__(self, 
                 env_name = "CarRacing-v3", 
                 render_mode = "rgb_array", 
                 lap_complete_percent = 0.95,
                 continuous = True,
                 framestack = 4):
        """
        TODO:
        """
        # Init Gym Env
        self.env = gym.make(env_name, 
                            render_mode = render_mode, 
                            lap_complete_percent = lap_complete_percent, 
                            domain_randomize = False, # HARDCODING as False for our purposes
                            continuous = continuous)
        #implemented grayscaling
        self.env = GrayscaleObservation(self.env, keep_dim=False)

        # 3. Resize: (96, 96) -> (84, 84) as described in the proposal. can be tweaked later if needed
        self.env = ResizeObservation(self.env, (84, 84))

        self.env = FrameStackObservation(self.env, stack_size=framestack)
        # Extract the Action and Observation Space from the created environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


    def reset(self, seed = None, options = None):
        '''
        TODO:
        '''
        observation, info = self.env.reset(seed = seed, options=options) #added options for compatibility
        return np.array(observation), info # convert to np.array for framestack


    def step(self,action):
        '''
        TODO:
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Invalid action!!')
        observation, reward, done, truncated, info = self.env.step(action)
        return np.array(observation), reward, done, truncated, info # convert to np.array for framestack


    def get_action_space(self):
        """
        TODO:
        """
        return self.action_space


    def get_observation_space(self):
        """
        TODO:
        """
        return self.observation_space


    def get_random_action(self):
        """
        TODO:
        """
        return self.action_space.sample()
    

    def render(self):
        """
        TODO:
        """
        return self.env.render()


    def close(self):
        '''
        TODO:
        '''
        self.env.close()