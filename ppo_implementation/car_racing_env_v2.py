import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from typing import Tuple
from gymnasium.wrappers import (
    FrameStackObservation, 
    GrayscaleObservation, 
    ResizeObservation
)

# Standard NO OP Action for the car
NO_OP_ACTION = np.array([0.0, 0.0, 0.0])


class CarRacingV3Wrapper(gym.Wrapper):
    def __init__(self,
                 args: Namespace,
                 env_name = "CarRacing-v3",
                 render_mode = None,
                 lap_complete_percent = 0.95,
                 continuous = True,
                 resize_shape: Tuple[int, int] = (84, 84)):
        """
        Constructor for initializing the CarRacingV3Wrapper.

        Args:
            env_name (str, optional): Represents the name of the env. Defaults to "CarRacing-v3".

            render_mode (str, optional): Represents the render mode. Defaults to None.

            lap_complete_percent (float, optional): Represents the percentage of tiles
                that must be visited by the agent before a lap is considered complete. Defaults to 0.95.

            continuous (bool, optional): Represents the condition to decide whether the agent
                uses continuous or discrete actions. Defaults to True.

            args (argparse.Namespace): Captures Special Arguments which must be provided.
        """
        # Init vanilla CarRacing-v3 env
        self.env = gym.make(env_name, 
                            render_mode = render_mode, 
                            lap_complete_percent = lap_complete_percent, 
                            continuous = continuous,
                            max_episode_steps = args.max_episode_steps * args.action_repetition + 100
                            )

        # Convert to Grayscale (96, 96, 3) -> (96, 96)
        self.env = GrayscaleObservation(self.env, keep_dim=False)

        # Resize the img dimensions to resize_shape (96, 96) -> (84, 84)
        # self.env = ResizeObservation(self.env, resize_shape)

        # Stack Frames (96, 96) -> (k, 96, 96)
        self.env = FrameStackObservation(self.env, stack_size=args.frame_stack_size)

        # Call the Parent Class Constructor
        super().__init__(self.env)

        # Store the arguments from command line
        self.action_repetition = args.action_repetition



    def reset(self, seed = None, options = None) -> np.ndarray:
        """
        Wrapper method on top of the original reset method. Performs following operations:
        - Skips first 50 zoom-in frames of the CarRacing-v3 Environment.
        - Converts the initial state (after zoom-in) into grayscale.
        - Stacks the grayscale initial state as a numpy array.

        Returns:
            np.ndarray: The initial state, stacked into frames.
        """
        # TODO: [If Needed] Implement a way to capture reward memory and initialize it here 

        # Reset the env to get the starting state
        state, info = self.env.reset(seed=seed, options = options) # state shape: (96, 96, 3)

        # Skip first 50 frames, as it involves zooming in by the env, 
        # which might confuse the Neural Network in order to learn
        # the right action for a given state
        for _ in range(50):
            state, _, _, _, info = self.env.step(NO_OP_ACTION)

        return state, info
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool]:
        """
        Wrapper method on top of original step method, which allows the agent to take an
        action in the current state and move to the next state. At a high level, it does
        the following:
        - Repeats the input action multiple times (according to action_repetition supplied for args in the constructor)
        - Accumulates the Total Reward after executing each action
        - Converts the final state after action repetition into grayscale
        - Pushes the final state into the stack frame

        Args:
            action (np.ndarray): The action represented by a numpy array of shape (3,), 
                which comes from the continuous action space.

        Returns:
            tuple (Tuple[np.ndarray, float, bool, bool]): Returns the stacked next state, total reward,
            done, and truncated after performing action repetition.
        """
        # TODO: Throw an error, if the action does not belong to the action space
        
        # Init variable to keep track of total_reward from this action
        total_reward = 0.0

        # Repeat action for the self.action_repetition times
        for _ in range(self.action_repetition):
            next_state, reward, done, truncated, info = self.env.step(action)

            # TODO: Implement Reward Augmentations here
            # Examples: Green Penalty, Die Penalty Removal etc.
            # self._compute_green_penalty(next_state_rgb)

            # Accumulate total reward for this action
            total_reward += reward

            # If episode has ended or truncated, stop action repetition
            if done or truncated:
                break

        return next_state, total_reward, done, truncated, info


    def render(self):
        """
        TODO:
        """
        # TODO: Implement this
        return self.env.render()
    

    def close(self):
        """
        TODO:
        """
        self.env.close()


    def _compute_green_penalty(self, img: np.ndarray) -> float:
        """
        TODO:

        Args:
            mg (np.ndarray): _description_

        Returns:
            float: _description_
        """

        #TODO: Compute penalty using wheel coordinates
        return 0.0
    

    def save_state_img(self, state: np.ndarray, img_title: str, img_file_name: str, cmap = None):
        """
        TODO:
        """
        if len(state.shape) == 3:
            img_to_show = state[-1]
        else:
            img_to_show = state

        plt.imshow(img_to_show, cmap = cmap)
        plt.title(img_title)
        plt.savefig(img_file_name)
        plt.close()  
