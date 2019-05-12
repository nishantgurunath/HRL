import gym
import numpy as np
import pickle
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp_policy import Policy
from models.mlp_policy_disc import DiscretePolicy


env = gym.make("FetchReach-v1")
is_disc_action = len(env.action_space.shape) == 0

policy_mgr, policy_wrk, _, _, _= pickle.load(open("../assets/learned_models/FetchReach-v1_a2c.p", "rb"))
state = env.reset()
total_reward = 0
steps = 0
for i in range(2000):
        env.render()
        # Manager part
        curr_pos = state['achieved_goal']
        desired_pos = state['desired_goal'] 
        state_var = np.concatenate((state['observation'],state['desired_goal']))
        state_mgr = torch.tensor(state_var).unsqueeze(0)
        subgoal = policy_mgr.select_action(state_mgr)[0]
        subgoal = subgoal.detach().numpy()
        state_wrk = torch.tensor(np.concatenate((state_var,subgoal))).unsqueeze(0)

        dist_mgr = np.linalg.norm(subgoal - desired_pos)
        dist_wrk = np.linalg.norm(subgoal - curr_pos)
        #print (dist_mgr)
        # Worker part
        while(dist_wrk > 1):
                action = policy_wrk.select_action(state_wrk)[0]
                # print(action.detach().numpy())
                state, reward, done, _ = env.step(action.detach().numpy())
                curr_pos = state['achieved_goal']
                state_wrk = np.concatenate((state['observation'],state['desired_goal'],subgoal))
                state_wrk = torch.tensor(state_wrk).unsqueeze(0)
                dist_wrk = np.linalg.norm(subgoal - curr_pos)
                total_reward += reward
                steps += 1
                print (dist_wrk)
                if done:
                        state = env.reset()
                        print(total_reward.item(), steps)
                        total_reward = 0
                        steps = 0
                        break


# env = gym.make("FetchReach-v1")
# state = env.reset()
# for i in range(1000):
#       # env.render()
#       action = env.action_space.sample()
#       state, reward, done, _ = env.step(action)
#       print(state['achieved_goal'])
#       if done: 
#               state = env.reset()
#               print(i)


