#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:29:38 2022

@author: guanfei1
"""


import torch
from torch import optim
from torch import nn
from collections import deque
import gym
import random
import numpy as np
class Actor(nn.Module):
    def __init__(self, device, env, hidden = 300, lr = 0.001, num_layers=3):
        super().__init__()
        info = []
        info.append(env.observation_space.shape[0], 
                                           hidden )
        for i in range(num_layers):
            if i != num_layers - 1:
                info.append(nn.Linear(hidden, hidden)
                info.append(nn.ReLU())
            else:
                info.append(nn.Linear(hidden, env.action_space.shape[0])
                info.append(nn.Tanh())
        self.net = nn.Sequential(*info)
            
        self.device=device
        self.tanh = nn.Tanh()
        self.optim = optim.Adam(self.parameters(), lr = lr)
    def forward(self, state):
        state=state.float()
        x = self.net(x)
        return x + torch.normal(mean=torch.tensor([0.]), 
                                std=torch.tensor([0.1])).to(self.device)
    
class Critic(nn.Module):
    def __init__(self, env, hidden=300, lr = 0.001, num_layers=3):
        super().__init__()
        info = []
        info.append(env.observation_space.shape[0] + env.action_space.shape[0], 
                                           hidden )
        for i in range(num_layers):
            if i != num_layers - 1:
                info.append(nn.Linear(hidden, hidden)
                info.append(nn.ReLU())
            else:
                info.append(nn.Linear(hidden, env.action_space.shape[0])
                info.append(nn.Identity())
        self.net = nn.Sequential(*info)
        self.optim = optim.Adam(self.parameters(), lr = lr)


    def forward(self, state, action):
        if len(action.shape) < len(state.shape):
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=1)
        return self.net(x)
  
class DDPG:
    def __init__(self, env, BATCH_SIZE):
        self.env = env
        self.device = torch.device("cuda" if\
        torch.cuda.is_available() else "cpu")
        self.actor = Actor(env=self.env,device=self.device).to(self.device)
        self.critic = Critic(env=self.env).to(self.device)
        self.target_actor = Actor(env=self.env,device=self.device).to(self.device)
        self.target_critic = Critic(env=self.env).to(self.device)
        self.gamma = 0.99
        self.tau = 0.005
        self.actor.load_state_dict(self.target_actor.state_dict())
        self.critic.load_state_dict(self.target_critic.state_dict())
        self.replay_buffer = deque(maxlen=1000000)
        self.loss = torch.nn.MSELoss()
        self.reward_buffer = deque(maxlen=100)
        self.action_high = torch.from_numpy(self.env.action_space.high).\
        to(self.device)
        self.count = 0
        self.batch_size=BATCH_sIZE

    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.to(self.device)
        
        return torch.clip(self.actor(state), -self.action_high, 
                          self.action_high).cpu().detach().tolist()
        
    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(),
                                       self.target_critic.parameters()):
         target_param.data.copy_(self.tau * param.data + \
                                 (1 - self.tau) * target_param.data)

         for param, target_param in zip(self.actor.parameters(), 
                        self.target_actor.parameters()):
             target_param.data.copy_(self.tau * param.data + 
                                (1 - self.tau) * target_param.data)
    def sample(self, batch_size = 100):
        t = random.sample(self.replay_buffer, self.batch_size)
        actions = []
        states = []
        dones = []
        states_ = []
        rewards = []
        for i in t:
            state, action, state_, reward, done  = i
            
            states.append(state)

            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            states_.append(state_)
        
        states = torch.from_numpy(np.array(states)).\
        to(self.device).float()
        actions = torch.from_numpy(np.array(actions)).\
        to(self.device).float()
        rewards = torch.from_numpy(np.array(rewards)).\
            to(self.device).float()
        dones = torch.from_numpy(np.array(dones)).\
            to(self.device)
        states_ = torch.from_numpy(np.array(states_)).to(self.device).float()
        
        return states, actions, states_, rewards, dones
        
    def train(self):
    
        states, actions, states_, rewards, dones = self.sample()
        target_action = self.target_actor(states_) + \
        torch.clip(torch.normal(torch.tensor([0.]), torch.tensor([0.2])), -0.5, 0.5).to(self.device)
        q_ = self.target_critic(states_, self.target_actor(states_)).view(-1)
        q_[dones]=0.
        with torch.no_grad(): 
           target_q = rewards + self.gamma * q_

        q = self.critic(states, actions).view(-1)
        critic_loss = self.loss(target_q,q)

        self.critic.optim.zero_grad()
        critic_loss.backward()
        self.critic.optim.step()
        if self.count % 2 == 0:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor.optim.zero_grad()
            actor_loss.backward()
            self.actor.optim.step()
            self.soft_update()
        
        self.count += 1

