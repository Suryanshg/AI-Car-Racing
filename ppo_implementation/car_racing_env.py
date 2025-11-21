import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Array to compute grayscale images from rgb images
STD_LUMINOSITY_FORMULA_ARR = [0.299, 0.587, 0.114]

# Standard NO OP Action for the car
NO_OP_ACTION = np.array([0.0, 0.0, 0.0])


class CarRacingV3Wrapper(gym.Wrapper):
    def __init__(self,
                 env_name = "CarRacing-v3",
                 render_mode = None,
                 lap_complete_percent = 0.95,
                 continuous = True,
                 args = None):
        """
        TODO:
        """
        # Init vanilla CarRacing-v3 env
        self.env = gym.make(env_name, 
                            render_mode = render_mode, 
                            lap_complete_percent = lap_complete_percent, 
                            continuous = continuous
                            )

        # Call the Parent Class Constructor
        super().__init__(self.env)

        # Store the arguments from command line
        self.action_repetition = args.action_repetition
        self.frame_stack_size = args.frame_stack_size

        # Init frames for stacking states
        self.frames = deque(maxlen = self.frame_stack_size)



    def reset(self):
        """
        TODO:
        """
        # TODO: Implement a way to capture reward memory and initialize it here

        # Reset the env to get the starting state
        state, info = self.env.reset() # state shape: (96, 96, 3)

        # Skip first 50 frames, as it involves zooming in by the env, 
        # which might confuse the Neural Network in order to learn
        # the right action for a given state
        for _ in range(50):
            state, _, _, _, _ = self.env.step(NO_OP_ACTION)
            

        # Convert RGB Image to Grayscale
        state_grayscale = self._rgb_to_grayscale(state) # state shape: (96, 96)

        # plt.imshow(state)
        # plt.title('RGB Image')
        # plt.savefig("rgb_img.png")
        # plt.close()  


        # plt.imshow(state_grayscale, cmap='gray')
        # plt.title('Grayscale Image')
        # plt.savefig("grayscale_img.png")
        # plt.close()    

        # TODO: Implement stacking states together
        # Clear all stacked frames
        self.frames.clear()

        # Stack initial state in the frames queue, "self.frame_stack_size" times
        for _ in range(self.frame_stack_size):
            self.frames.append(state_grayscale)
        stacked_frames = np.stack(self.frames, axis = 0) # (frame_stack_size, 96, 96)

        return stacked_frames
    

    def step(self, action):
        """
        TODO:
        """
        # TODO: Improve this method
        if not self.env.action_space.contains(action):
            raise ValueError('Invalid action!!')
        observation, reward, done, truncated, info = self.env.step(action)
        return observation, reward, done, truncated, info


    def render(self):
        """
        TODO:
        """
        # TODO: Implement this
        pass



    def _rgb_to_grayscale(self, rgb_img):
        """
        TODO:
        """
        grayscale_img = np.dot(rgb_img[..., :3], STD_LUMINOSITY_FORMULA_ARR)

        return grayscale_img