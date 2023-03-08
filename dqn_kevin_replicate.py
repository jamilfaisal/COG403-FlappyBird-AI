# https://cs229.stanford.edu/proj2015/362_report.pdf

import torch
import time
import flappy_bird_gym

env = flappy_bird_gym.make("FlappyBird-v0")
env_rgb = flappy_bird_gym.make("FlappyBird-rgb-v0")

obs = env.reset()
obs_rgb = env_rgb.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()  # env.action_space.sample() for a random action

    # Processing:
    obs, reward, done, info = env.step(action)
    info  = env_rgb.step(action)
    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS

    # Checking if the player is still alive
    if done:
        break

env.close()