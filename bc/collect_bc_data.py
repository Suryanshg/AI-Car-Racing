from environment_framestacking import CarRacingV3Wrapper
import gymnasium as gym
import pygame
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import sys
import random
import os
import string
import matplotlib.pyplot as plt


NO_OP = np.array([0.0, 0.0, 0.0], dtype=np.float32)
GAME_FPS = 50


def get_action(keys, add_noise=False):
    steer = 0
    if keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_RIGHT]:
        steer += 1.0
    gas = 1.0 if keys[pygame.K_UP] else 0
    brake = 1.0 if keys[pygame.K_DOWN] else 0
    if add_noise:
        steer = steer + np.random.normal(0, 0.05)
        gas = gas + np.random.normal(0, 0.05)
        brake = brake + np.random.normal(0, 0.05)
        steer = np.clip(steer, -1.0, 1.0)
        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
    return np.array([steer, gas, brake], dtype=np.float32)


if __name__ == "__main__":
    # get name from cmd else prompt
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = input("Enter name: ")
    name = name.lower()
    save_folder = f"./bc-logs/{name}"
    os.makedirs(save_folder, exist_ok=True)

    pygame.init()
    # dummy window just to grab keyboard input
    screen = pygame.display.set_mode((600, 400))

    # load env wrapper
    env = CarRacingV3Wrapper(
        env_name="CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.9,
        continuous=True,
        framestack=4,
    )
    obs, _ = env.reset()
    total_reward = 0.0

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
        screen.blit(
            pygame.surfarray.make_surface(display_obs.transpose((1, 0, 2))), (0, 0)
        )
        pygame.display.update()

        keys = pygame.key.get_pressed()
        action = get_action(keys)
        data_obs.append(obs)
        data_act.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        length += 1
        print(length, obs.shape, reward, total_reward, "ACTION:", action)
        if terminated:
            print("")
            print("Episode ended!", terminated)
            running = False

        clock.tick(int(GAME_FPS))  # match environment fps

    env.close()
    pygame.quit()
    # to save or not to save
    # total possible length
    if length < 950 and terminated:
        save_run = True
        output_folder = f"{save_folder}/excellent_runs/"
    else:
        save_run = True
        if total_reward > 800:
            output_folder = f"{save_folder}/800_runs/"
        elif total_reward > 700:
            output_folder = f"{save_folder}/700_runs/"
        else:
            save_run = False
    if save_run and terminated and total_reward > 700:
        os.makedirs(output_folder, exist_ok=True)
        # save data with name in filename
        id = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        output_filename = output_folder + f"{id}_bc.npz"
        np.savez(output_filename, obs=np.array(data_obs), actions=np.array(data_act))
        print(f"Data saved to {output_filename}")
        print(f"Total observations collected: {len(data_obs)}")
        print(f"Observation shape: {np.array(data_obs).shape}")
    else:
        print("Run not saved due to insufficient length/reward.")
