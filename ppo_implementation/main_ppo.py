import argparse
import time
from arguments import get_args
# from car_racing_env import CarRacingV3Wrapper
from car_racing_env_v2 import CarRacingV3Wrapper
from agent_ppo import Agent_PPO
from eval_ppo import eval_ppo
from gymnasium.wrappers import FrameStackObservation

def parse():
    """
    TODO:
    """
    # Initialize an Argument Parser
    parser = argparse.ArgumentParser(description = "PPO Agent for CarRacing-v3")

    # Add Initial Arguments
    parser.add_argument('--train_ppo', action='store_true', help='whether to train PPO')
    parser.add_argument('--test_ppo', action='store_true', help='whether to test PPO')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing')

    # Fetch Special Arguments
    parser = get_args(parser)

    # Parse all the Arguments
    args = parser.parse_args()

    # Return the parser arguments
    return args


def run(args):
    """
    TODO: 
    """
    # Start the timer
    start_time = time.time()

    # If user requested training mode
    if args.train_ppo:

        # Intialize the Environment
        env = CarRacingV3Wrapper(args=args)

        # Stack Frames (96, 96) -> (4, 96, 96)
        env = FrameStackObservation(env, stack_size=args.frame_stack_size)

        # Initialize the PPO Agent
        agent = Agent_PPO(env, args)

        # Train the PPO Agent
        agent.train()

        # Perform Test Drive
        # agent.test_drive()

    # If user requested testing mode
    if args.test_ppo:

        # Determine render mode value
        # TODO: Check if we need to use human here instead of rgb_array
        render_mode_value = "rgb_array" if args.record_video else None
        
        # Initialize Environment using the determined render_mode_value
        env = CarRacingV3Wrapper(render_mode=render_mode_value, args=args)

        # Stack Frames (96, 96) -> (4, 96, 96)
        env = FrameStackObservation(env, stack_size=args.frame_stack_size)

        # Initialize the PPO Agent
        agent = Agent_PPO(env, args)

        # Run the evaluation method
        eval_ppo(agent, env, total_episodes=10, record_video=args.record_video)

    # Stop the timer and print the runtime
    print(f"Total Runtime: {time.time() - start_time:.4f}s")


# Driver Code
if __name__ == '__main__':
    args = parse()
    run(args)

