import gymnasium as gym
import pygame
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import sys
import random
import os
import string


NO_OP = np.array([0.0, 0.0, 0.0], dtype=np.float32)
GAME_FPS = 50


def get_action(keys):
    steer = 0
    if keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_RIGHT]:
        steer += 1.0
    gas = 1.0 if keys[pygame.K_UP] else 0.0
    brake = 1.0 if keys[pygame.K_DOWN] else 0.0
    return np.array([steer, gas, brake], dtype=np.float32)


if __name__ == "__main__":
    # get name from cmd else prompt
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = input("Enter name: ")
    name = name.lower()
    save_folder = f'bc-logs/{name}'
    os.makedirs(save_folder, exist_ok=True)

    pygame.init()
    # dummy window just to grab keyboard input
    screen = pygame.display.set_mode((600, 400))

    # load env
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    obs, _ = env.reset()
    total_reward = 0.0
    # init frame stack to store past 4 consecutive observations
    frame_stack = deque(maxlen=4)
    # add first observation
    frame_stack.append(obs)

    length = 0
    data_obs = []
    data_act = []
    save_run = False
    done = False
    running = True
    clock = pygame.time.Clock()
    # loop while running or the episode is not done
    while running:
        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        display_obs = env.render()
        # render to screen
        screen.blit(pygame.surfarray.make_surface(
            display_obs.transpose((1, 0, 2))), (0, 0))
        pygame.display.update()

        keys = pygame.key.get_pressed()
        action = get_action(keys)

        obs, reward, terminated, truncated, info = env.step(action)
        # add current observation to frame stack
        frame_stack.append(obs)
        total_reward += reward
        length += 1
        print(length, len(frame_stack))
        if len(frame_stack) == 4:
            stacked_obs = np.array(list(frame_stack))
            data_obs.append(stacked_obs)
            data_act.append(action)
        if terminated or truncated:
            running = False

        clock.tick(GAME_FPS)  # match environment fps

    env.close()
    pygame.quit()
    print(length, total_reward)
    # to save or not to save
    if length < 1000:
        save_run = True
        output_folder = f"{save_folder}/excellent_runs/"
    else:
        save_run = True
        done = True
        if total_reward > 800:
            output_folder = f"{save_folder}/800_runs/"
        elif total_reward > 700:
            output_folder = f"{save_folder}/700_runs/"
        else:
            save_run = False
    if save_run and truncated:
        os.makedirs(output_folder, exist_ok=True)
        # save data with name in filename
        id = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=4))
        output_filename = output_folder + f"{id}_bc.npz"
        np.savez(output_filename, obs=np.array(
            data_obs), actions=np.array(data_act))
        print(f"Data saved to {output_filename}")
        print(f"Total observations collected: {len(data_obs)}")
        print(
            f"Observation shape: {np.array(data_obs).shape}")
    else:
        print("Run not saved due to insufficient length/reward.")
