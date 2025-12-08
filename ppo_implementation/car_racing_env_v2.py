from collections import deque

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
import time

# Standard NO OP Action for the car
NO_OP_ACTION = np.array([0.0, 0.0, 0.0])

# Constants for the Car's position in the 96x96 observation
# The camera is locked to the car, so these never change.
# We look slightly below the center to check the pixels under the tires.
CAR_H_START = 67
CAR_H_END = 77
CAR_W_START = 45
CAR_W_END = 51

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
        # self.env = GrayscaleObservation(self.env, keep_dim=False)

        # Resize the img dimensions to resize_shape (96, 96) -> (84, 84)
        # self.env = ResizeObservation(self.env, resize_shape)

        # Stack Frames (96, 96) -> (k, 96, 96)
        # self.env = FrameStackObservation(self.env, stack_size=args.frame_stack_size)

        # Call the Parent Class Constructor
        super().__init__(self.env)

        # Manually Override the observation space to tell that a grayscale img will be returned (with 1 channel only)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(96, 96), 
            dtype=np.uint8
        )

        # Store the arguments from command line
        self.action_repetition = args.action_repetition
        self.gas_promotion = deque(maxlen=5)



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
        state_rgb, info = self.env.reset(seed=seed, options = options) # state shape: (96, 96, 3)

        # Skip first 50 frames, as it involves zooming in by the env, 
        # which might confuse the Neural Network in order to learn
        # the right action for a given state
        for _ in range(50):
            state_rgb, _, _, _, info = self.env.step(NO_OP_ACTION)

        # Convert RGB (96, 96, 3) to Grayscale (96, 96)
        # Luminance formula: 0.299 R + 0.587 G + 0.114 B
        state_gray = np.dot(state_rgb[..., :3], [0.299, 0.587, 0.114])
        state_gray = state_gray.astype(np.uint8)

        return state_gray, info
    

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
            next_state_rgb, reward, done, truncated, info = self.env.step(action)

            # TODO: Implement Reward Shaping here
            # Examples: Green Penalty, Die Penalty Removal etc.

            # Compute and subtract green penalty from the reward
            reward -= self._compute_green_penalty(next_state_rgb)

            # Add reward for prioritizing gas in last 5 actions
            self.gas_promotion.append(np.clip(action[1], 0, 1))
            reward += np.sum(np.array(self.gas_promotion))/100. # Max gas reward = 1*1000/100 = 10

            # Accumulate total reward for this action
            total_reward += reward

            # If episode has ended or truncated, stop action repetition
            if done or truncated:
                break

        # Convert RGB (96, 96, 3) to Grayscale (96, 96)
        # Luminance formula: 0.299 R + 0.587 G + 0.114 B
        next_state_gray = np.dot(next_state_rgb[..., :3], [0.299, 0.587, 0.114])
        next_state_gray = next_state_gray.astype(np.uint8)

        return next_state_gray, total_reward, done, truncated, info


    def _compute_green_penalty(self, img: np.ndarray) -> float:
        """
        TODO:

        Args:
            mg (np.ndarray): _description_

        Returns:
            float: _description_
        """

        # Crop the area under the car tires
        # patch = img[CAR_H_START:CAR_H_END, CAR_W_START:CAR_W_END]
        patch = img[62:80, 40:56]

        # save_state_img(patch, "Patch", "patch.png")
        
        # Compute the average color of that patch
        # Axis (0, 1) averages the height and width, leaving the 3 RGB channels
        mean_color = patch.mean(axis=(0, 1))
        
        r = mean_color[0]
        g = mean_color[1]
        b = mean_color[2]
        
        # print(f"\nr: {r:.4f}, g: {g:.4f}, b: {b:.4f}")

        # Detect Green Dominance over Red
        if g > r + 15:

            # The car is on the grass!
            # print("Car on grass!")
            # time.sleep(5)

            # Max grass penalty ~g*950
            return 0.05  # Return the penalty amount
        
        # The car is on the road (or red/white curb)
        return 0.0


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


def save_state_img(state: np.ndarray, img_title: str, img_file_name: str, cmap = None):
    """
    TODO:
    """
    # Case 1: RGB Image (Height, Width, 3)
    # We want to show the whole thing, not slice it.
    if len(state.shape) == 3 and state.shape[-1] == 3:
        img_to_show = state
        # RGB images don't use a colormap (like 'gray')
        cmap = None 

    # Case 2: Frame Stack (Stack_Size, Height, Width)
    # We want to show the most recent frame (the last one).
    elif len(state.shape) == 3:
        img_to_show = state[-1]
        
    # Case 3: Grayscale Image (Height, Width)
    else:
        img_to_show = state

    plt.imshow(img_to_show, cmap=cmap)
    plt.title(img_title)
    plt.savefig(img_file_name)
    plt.close()
