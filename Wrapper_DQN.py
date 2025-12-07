
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
)
from gymnasium.wrappers import TransformObservation


class CarRacingV3Wrapper(gym.Wrapper):

    def __init__(self, render_mode="rgb_array"):
        base_env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
        )

        super().__init__(base_env)


class DiscreteActionWrapper(gym.ActionWrapper):

    def __init__(self, env, num_actions=10):
        super().__init__(env)

        self.actions = np.array([
            [0.0, 0.0, 0.0],      # 0 no-op
            [0.0, 1.0, 0.0],      # 1 gas
            [0.0, 0.0, 1.0],      # 2 brake
            [-1.0, 0.6, 0.0],     # 3 left mild
            [1.0, 0.6, 0.0],      # 4 right mild
            [-0.5, 1.0, 0.0],     # 5 left strong
            [0.5, 1.0, 0.0],      # 6 right strong
            [-1.0, 0.0, 0.8],     # 7 left brake
            [1.0, 0.0, 0.8],      # 8 right brake
            [0.0, 0.5, 0.5],      # 9 slow forward
        ], dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]


class IgnoreFirstNFrames(gym.Wrapper):
    def __init__(self, env, n=50):
        super().__init__(env)
        self.n = n

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for _ in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                return self.reset(**kwargs)

        return obs, info


def make_env(render_mode=None):

    env = CarRacingV3Wrapper(render_mode=render_mode)

    env = DiscreteActionWrapper(env)

    env = GrayscaleObservation(env, keep_dim=False)

    env = ResizeObservation(env, (84, 84))

    env = FrameStackObservation(env, stack_size=4)

    env = IgnoreFirstNFrames(env, n=50)

    env = TransformObservation(
        env,
        lambda obs: obs.astype(np.float32) / 255.0
    )

    return env
