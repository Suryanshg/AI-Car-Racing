from stable_baselines3 import A2C
from environment import CarRacingV3Wrapper

# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

# Import the CarRacingV3Wrapper
env = CarRacingV3Wrapper()


# Initialize A2C Implementation for SB3
model = A2C("CnnPolicy", env, verbose=1, device = "cpu")
model.learn(total_timesteps=2500)
model.save("a2c_car_racing")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_car_racing", device = "cpu")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    env.render()