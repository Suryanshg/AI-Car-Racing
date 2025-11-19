import gymnasium as gym

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


    def reset(self):
        """
        TODO:
        """
        # TODO: Implement a way to capture reward memory and initialize it here

        # Reset the env to get the starting state
        state = self.env.reset() # state shape: (96, 96, 3)

        # TODO: Convert RGB Image to Grayscale

        # TODO: Implement stacking states together

        return state
    

    def step(self, action):
        """
        TODO:
        """
        # TODO: Implement this
        pass


    def render(self):
        """
        TODO:
        """
        # TODO: Implement this
        pass



    def _rgb_to_grayscale(rgb_img):
        """
        TODO:
        """