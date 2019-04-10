import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_memory import ReplayMemory
from utils import plotting

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def one_hot_action(action):
    vector = np.zeros(81)
    vector[action] = 1.0
    return np.expand_dims(vector, axis=0)

def one_hot_goal(goal):
    vector = np.zeros(81)
    vector[goal] = 1.0
    return np.expand_dims(vector, axis=0)

def hdqn_learning(
    env,
    agent,
    num_episodes,
    exploration_schedule,
    gamma=1.0,
    ):

    """The h-DQN learning algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    agent:
        a h-DQN agent consists of a meta-controller and controller.
    num_episodes:
        Number (can be divided by 1000) of episodes to run for. Ex: 12000
    exploration_schedule: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    gamma: float
        Discount Factor
    """
    ###############
    # RUN ENV     #
    ###############
    # Keep track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    total_timestep = 0
    meta_timestep = 0

    for i_thousand_episode in range(1):
        for i_episode in range(num_episodes):
            episode_length = 0
            current_state = env.reset()
           
            done = False
            while not done:
                meta_timestep += 1
                # Get annealing exploration rate (epislon) from exploration_schedule
                meta_epsilon = exploration_schedule.value(total_timestep)
                goal = agent.select_goal(current_state.reshape(1,-1), meta_epsilon)[0]
                encoded_goal = one_hot_goal(goal)

                total_extrinsic_reward = 0
                goal_reached = False
                s1 = current_state.reshape(1,-1)
                while not done and not goal_reached:
                #while not done:
                    total_timestep += 1
                    episode_length += 1
                    # Get annealing exploration rate (epislon) from exploration_schedule
                    ctrl_epsilon = exploration_schedule.value(total_timestep)
                    joint_state_goal = np.concatenate((current_state.reshape(1,-1), encoded_goal),axis=1)
                    #joint_state_goal = current_state.reshape(1,-1)
                    action = agent.select_action(joint_state_goal, ctrl_epsilon)[0]
                    action_x, action_y = agent.idx_2_action[int(action)]
                    ### Step the env and store the transition
                    next_state, extrinsic_reward, done, _ = env.step((action_y,action_x))
                    # Update statistics
                    stats.episode_rewards[i_thousand_episode*1000 + i_episode] += extrinsic_reward
                    stats.episode_lengths[i_thousand_episode*1000 + i_episode] = episode_length

                    intrinsic_reward = agent.get_intrinsic_reward(goal, (next_state[4],next_state[5]))
                    goal_reached = agent.get_quadrant(next_state[4],next_state[5]) == (goal)

                    joint_next_state_goal = np.concatenate((next_state.reshape(1,-1), encoded_goal),axis=1)
                    #joint_next_state_goal = next_state.reshape(1,-1)
                    agent.ctrl_replay_memory.push(joint_state_goal, action, joint_next_state_goal, intrinsic_reward, done)
                    #agent.ctrl_replay_memory.push(joint_state_goal, action, joint_next_state_goal, extrinsic_reward, done)
                    # Update Both meta-controller and controller
                    agent.update_meta_controller(gamma)
                    agent.update_controller(gamma)
                    agent.update_target()
            
                    total_extrinsic_reward += extrinsic_reward
                    current_state = next_state.reshape(1,-1)
                # Goal Finished
                agent.meta_replay_memory.push(s1, goal, next_state.reshape(1,-1), total_extrinsic_reward, done)

    return agent, stats