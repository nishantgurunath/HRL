import numpy as np
import gym
import random
#import time
#import pandas as pd
import matplotlib.pyplot as plt

# env = gym.make("Reacher-v2")

# env.reset()
# for i in range(10):
#     obs, rew, done, _ = env.step(env.action_space.sample())
#     print(rew, done)

# env.close()

rewards1 = np.load('mdp_20000_raw.npy')
# rewards_smoothed = pd.Series(rewards).rolling(10, min_periods=10).mean()
rewards1 = rewards1.reshape(-1,500)
rewards1 = np.mean(rewards1, axis=1)
print (rewards1.shape)

rewards2 = np.load('n20000.npy')
rewards2 = rewards2.reshape(-1,500)
rewards2 = np.mean(rewards2, axis=1)
# rewards_smoothed = pd.Series(rewards).rolling(10, min_periods=10).mean()



plt.title("Rewards(avg over 500 episodes) for h-DQN vs DQN on MDP env")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(rewards1, label = 'DQN')
plt.plot(rewards2, label = 'h-DQN')
plt.legend()
plt.savefig("rewards")
#plt.show()
#plt.clf()

