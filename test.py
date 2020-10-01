import gym
import random

import numpy as np


env = gym.make('Breakout-ram-v0')
env.reset()

obs_set = set()
for ep in range(10):
	observation = env.reset()
	for t in range(1000):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		obs_set = obs_set|set(observation)
		if reward > 0:
			print(action, reward)
		if done:
			print(f"Episode finished after {t + 1} timesteps.")
			break 		

env.close()
