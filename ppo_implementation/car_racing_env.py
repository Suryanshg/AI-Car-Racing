'''
# For algorithms like PPO with continuous actions
env_continuous = CarRacingV3Wrapper(continuous=True) #call the wrapper as continuous
action = np.array([0.5, 0.8, 0.0])  # Example continuous action
obs, reward, done, truncated, info = env_continuous.step(action) #step in the env

# For algorithms like DQN that need discrete actions
env_discrete = CarRacingV3Wrapper(continuous=False) #call the wrapper as discrete
action = 3  # Example discrete action
obs, reward, done, truncated, info = env_discrete.step(action) #step in the env

# Additionally, you can create a preset of arguments for your model.

EXAMPLE ARGUMENTS FOR A DISCRETE ACTION ENVIRONMENT WITH FRAMESTACKING AND ACTION REPETITION:
def get_args():

    args = argparse.Namespace()

    args.continuous = False 
    
    args.framestack = 4
    
    args.action_repetition = 8 
    
    args.max_episode_steps = 600
    
    args.resize_shape = (84, 84)

    return args

Feed the args object to the wrapper:
env = CarRacingV3Wrapper(args=args)

You can also feed arguments directly if desired
'''


import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
import numpy as np
from typing import Tuple, Optional
from argparse import Namespace

CONTINUOUS_NO_OP = np.array([0.0, 0.0, 0.0], dtype=np.float32)
DISCRETE_NO_OP = 0

class CarRacingV3Wrapper(gym.Wrapper):
    """
    Customizable Environment for the CarRacing-v3 env from Gymnasium.
    """

    def __init__( #overhaul init method, many new feedable params
        self,
        args: Optional[Namespace] = None,
        env_name: str = "CarRacing-v3",
        render_mode: Optional[str] = "rgb_array",
        lap_complete_percent: float = 0.95,
        continuous: bool = True,
        framestack: int = 4,
        action_repetition: int = 1,
        resize_shape: Tuple[int, int] = (84, 84),
        max_episode_steps: Optional[int] = None,
    ):
        """
        Initialize the CarRacing environment with preprocessing wrappers.
        
        Args:
            args: Optional Namespace with config. If provided, overrides other params.
                  Expected fields: frame_stack_size, action_repetition, max_episode_steps
                  
            env_name: Name of the Gymnasium environment
            
            render_mode: How to render ('rgb_array', 'human', None)
            
            lap_complete_percent: Fraction of track tiles needed to complete lap
            
            continuous: If True, use continuous actions. If False, use discrete.
            
            framestack: Number of frames to stack for temporal information
            
            action_repetition: Number of times to repeat each action
            
            resize_shape: Target size for observations (height, width)
            
            max_episode_steps: Max steps per episode. None uses env default.
        """
        # If args provided, override individual parameters
        if args is not None:
            framestack = getattr(args, 'framestack', framestack)
            action_repetition = getattr(args, 'action_repetition', action_repetition)
            max_episode_steps = getattr(args, 'max_episode_steps', max_episode_steps)
        # Store configuration    
        self.continuous = continuous
        self.action_repetition = action_repetition
        self.no_op_action = CONTINUOUS_NO_OP if continuous else DISCRETE_NO_OP
        
        
        # Init Gym Env
        # Build the environment with optional max_episode_steps
        # Adding buffer to max_episode_steps to account for skipped zoom frames
        env_kwargs = {
            "render_mode": render_mode,
            "lap_complete_percent": lap_complete_percent,
            "domain_randomize": False,
            "continuous": continuous,
        }
        if max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = max_episode_steps + 100
        
        env = gym.make(
            env_name,
            **env_kwargs
        )

    
        # Greyscale: (96, 96, 3) -> (96, 96)
        env = GrayscaleObservation(env, keep_dim=False)
 
        # TODO: Can we make it explicit?
        # Resize: (96, 96) -> resize_shape
        # env = ResizeObservation(env, resize_shape)
        
        # FrameStack: (resize_shape) -> (k, resize_shape)
        env = FrameStackObservation(env, stack_size=framestack)
        
        super().__init__(env) #call parent constructor
        

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        
        
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Skip 50 frames to get past the zoom-in animation
        for _ in range(50):
            observation, _, _, _, _ = self.env.step(self.no_op_action)
            
        return np.array(observation), info


    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
            Take a step in the environment with action repetition.
        """
        if not self.env.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. "
                f"Expected action in {self.action_space}")
        
        total_reward = 0.0 #accumulate reward over action repetitions
        
        for _ in range(self.action_repetition):
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
        
            if done or truncated:
                break
        
        return (
            np.array(observation),
            total_reward,
            done,
            truncated,
            info,
        )  # convert to np.array for framestack

    def get_action_space(self):
        """
            Get the action space of the environment.
        """
        return self.action_space

    def get_observation_space(self):
        """
            Get the observation space of the environment.
        """
        return self.observation_space

    def get_random_action(self):
        """
            Get a random action from the action space.
        """
        return self.action_space.sample()

    def render(self):
        """
            Render the environment.
        """
        return self.env.render()
    
    def is_continuous(self):
        """
            True if continuous actions, False if discrete
        """
        return self.continuous

    def close(self):
        """
            Close the environment.
        """
        self.env.close()