#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:04:16 2022

@author: guanfei1
"""

import gym
import numpy as np
from gym import wrappers
from DDPG import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v4', render_mode='human')
    agent = DDPG(env)
    ys = []
    eps = 10000
    xs = list(range(eps))
    print("Replay Buffer Initialized")
    for j in range(eps):
        done = False
        episode_reward = 0
        s = env.reset()[0]
        while not done:
            action = agent.act(s)
            s_, r, done, _, _ = env.step(action)
            agent.replay_buffer.append((s, action, s_, r, done))
            s = s_
            episode_reward += r
            if agent.batch_size < len(agent.replay_buffer):
                agent.train()
        agent.reward_buffer.append(episode_reward)
        mean = np.mean(agent.reward_buffer)
        ys.append(mean)
        
        print("Episode Reward", episode_reward, ", Average Reward", 
              mean)
    plt.plot(xs, ys)
    
    

