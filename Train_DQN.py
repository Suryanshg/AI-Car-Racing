# train_dqn.py
from stable_baselines3 import DQN
from environment_framestacking import CarRacingV3Wrapper
from gpu_cleanup_callback import GPUCleanupCallback
import torch

env = CarRacingV3Wrapper(
    continuous=False,   # <-- ONLY CHANGE NEEDED FOR YOU
    framestack=4,
    resize_shape=(84, 84),
    action_repetition=1,
)

callback = GPUCleanupCallback(interval=5000)
model = DQN(
    "CnnPolicy",
    env,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=32,
    learning_rate=1e-4,
    train_freq=4,
    target_update_interval=8_000,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


model.learn(total_timesteps=1_000_000, callback=callback)
model.save("dqn_car_racing")
env.close()
