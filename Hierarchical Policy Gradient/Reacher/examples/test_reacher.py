import gym
import numpy as np
import pickle
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp_policy import Policy
from models.mlp_policy_disc import DiscretePolicy
from core.agent import get_target

env = gym.make("Reacher-v2")
is_disc_action = len(env.action_space.shape) == 0

policy_mgr, policy_wrk, _, _, _= pickle.load(open("../assets/learned_models/Reacher-v2_trpo.p", "rb"))
state,curr_pos = env.reset()
total_reward = 0
steps = 0
direction = policy_mgr.select_action(torch.tensor(state).unsqueeze(0))[0]
direction = int(direction.detach().numpy())
subgoal = get_target(curr_pos,direction)
#done_count = 0
for i in range(2000):

    env.render()
    state_wrk = np.concatenate((state,subgoal))
    action = policy_wrk.select_action(torch.tensor(state_wrk).unsqueeze(0))[0]
    state, reward, done, info = env.step(action.detach().numpy())
    #print (done_count)
    #if(done): done_count += 1
    #print (done_count)
    #done = (done_count == 1000)
    total_reward += reward
    steps += 1

    reward_wrk = - np.linalg.norm(subgoal - info['fingertip'])
    subgoal_reached = (-reward_wrk < 0.15)    

    #print ('W:',-reward_wrk)

    if(subgoal_reached):
        direction = policy_mgr.select_action(torch.tensor(state).unsqueeze(0))[0]
        direction = int(direction.detach().numpy())
        curr_pos = info['fingertip']
        subgoal = get_target(curr_pos,direction)
        #print('Manager:',np.linalg.norm(subgoal - info['target']))
        #print ("     ")

    if done:
        state,curr_pos = env.reset()
        #done_count = 0
        direction = policy_mgr.select_action(torch.tensor(state).unsqueeze(0))[0]
        direction = int(direction.detach().numpy())
        subgoal = get_target(curr_pos,direction)
        print(total_reward, steps)
        #print("   ")
        #print('Manager:',np.linalg.norm(subgoal - info['target']))
        total_reward = 0
        steps = 0
        #break


#env,_ = gym.make("Reacher-v2")
#state = env.reset()
#for i in range(1000):
#      # env.render()
#      action = env.action_space.sample()
#      state, reward, done, info = env.step(action)
#      print(info['reward_dist'])
#      if done: 
#              state,_ = env.reset()
#              print(i)


