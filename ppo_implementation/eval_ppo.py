from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import time
import numpy as np


def eval_ppo(agent, env, total_episodes = 10, record_video = False):
    """
    TODO:

    Args:
        agent (_type_): _description_
        env (_type_): _description_
        total_episodes (int, optional): _description_. Defaults to 10.
        record_video (bool, optional): _description_. Defaults to False.
    """

    # Initialize a list to keep track of rewards gained per episode
    rewards = []

    # TODO: Implement this
    # env.seed(seed)

    # If recording of the video was requested
    if record_video:
       
       # Wrap the env using RecordVideo wrapper
       env = RecordVideo(
       env,
       video_folder="./videos", # Save location folder for the videos
       name_prefix="test_vid", # Name of the prefix of each video file
       episode_trigger=lambda ep_id: True  # record video for every episode
    )
    
    # Record the start time
    start_time = time.time()
    
    # Iterate over total_episodes times
    for _ in tqdm(range(total_episodes)):

        # Init reward for this episode
        episode_reward = 0.0

        # Init the env and get the first state
        state = env.reset()

        # Flags for checking if the episode is over
        truncated = False
        done = False

        # If the episode is not over yet
        while not done and not truncated:

            # Select an action using the trained agent
            action, _, _ = agent.select_action(state)

            # Take the step using the action in the env and observe the reward and next state
            state, reward, done, truncated = env.step(action)

            # Accumulate the reward
            episode_reward += reward
    
        # Add the reward for this episode in the list
        rewards.append(episode_reward)

    # Safely close the env
    env.close()

    # Print the statistics
    print(f'Ran evaluation for {total_episodes} episodes')
    print('Mean Rewards:', np.mean(rewards))
    print('Rewards List:', rewards)
    print('Running time', time.time() - start_time)