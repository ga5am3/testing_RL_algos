import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Create and wrap the environment
env = make_vec_env("Pendulum-v1", n_envs=1)

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_pendulum_tensorboard/")

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("sac_pendulum")

# Load the trained model
model = SAC.load("sac_pendulum")

# Test the trained model
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

env.close()