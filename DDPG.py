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
import stochastic_process
import numpy as np
class Actor(nn.Module):
    def __init__(self, env, hidden = 300, lr = 0.0001):
        super().__init__()
        self.linear1 = nn.Linear(env.observation_space.shape[0], 
                                           hidden + 100)
        f1 = 1./np.sqrt(self.linear1.weight.data.size()[0])
        nn.init.uniform_(self.linear1.weight.data, -f1, f1)
        nn.init.uniform_(self.linear1.bias.data, -f1, f1)

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100, hidden)
        self.linear3 = nn.Linear(hidden, env.action_space.shape[0])

        self.tanh = nn.Tanh()
        f2 = 1./np.sqrt(self.linear2.weight.data.size()[0])
        nn.init.uniform_(self.linear2.weight.data, -f2, f2)
        nn.init.uniform_(self.linear2.bias.data, -f2, f2)
        f3 = 0.003
        nn.init.uniform_(self.linear3.weight.data, -f3, f3)
        nn.init.uniform_(self.linear3.bias.data, -f3, f3)
        self.optim = optim.Adam(self.parameters(), lr = lr)
        self.stochastic_process = stochastic_process.OrnsteinUhlenbeckProcess()
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x + torch.tensor(self.stochastic_process.sample(),
                                dtype=torch.float32).cuda()
    def train(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    
class Critic(nn.Module):
    def __init__(self, env, hidden=400, lr = 0.0001):
        super().__init__()
        self.linear1= nn.Linear(env.observation_space.shape[0], 
                                           hidden + 100)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100 + env.action_space.shape[0], hidden)
                                 
        self.linear3 = nn.Linear(hidden, 1)
        self.optim = optim.Adam(self.parameters(), lr = lr)
        f1 = 1./np.sqrt(self.linear1.weight.data.size()[0])
        nn.init.uniform_(self.linear1.weight.data, -f1, f1)
        nn.init.uniform_(self.linear1.bias.data, -f1, f1)
        f2 = 1./np.sqrt(self.linear2.weight.data.size()[0])
        nn.init.uniform_(self.linear2.weight.data, -f2, f2)
        nn.init.uniform_(self.linear2.bias.data, -f2, f2)
        f3 = 1./np.sqrt(self.linear3.weight.data.size()[0])
        nn.init.uniform_(self.linear3.weight.data, -f3, f3)
        nn.init.uniform_(self.linear3.bias.data, -f3, f3)


    def forward(self, state, action):
        if len(action.shape) < len(state.shape):
            action = action.unsqueeze(-1)
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x, action], dim=1))
        x = self.relu(x)
        x = self.linear3(x)

        return x
  
class DDPG:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if\
        torch.cuda.is_available() else "cpu")
        self.actor = Actor(env=self.env).to(self.device)
        self.critic = Critic(env=self.env).to(self.device)
        self.target_actor = Actor(env=self.env).to(self.device)
        self.target_critic = Critic(env=self.env).to(self.device)
        self.gamma = 0.99
        self.tau = 0.001
        self.actor.load_state_dict(self.target_actor.state_dict())
        self.critic.load_state_dict(self.target_critic.state_dict())
        self.replay_buffer = deque(maxlen=1000000)
        self.loss = torch.nn.MSELoss()
        self.reward_buffer = deque(maxlen=1000)
        self.action_high = torch.from_numpy(self.env.action_space.high).\
        to(self.device)
        s = env.reset()
        for i in range(100):
            done = False
            while not done:
                action = self.env.action_space.sample()
                s_, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((s, action, s_, reward, done))
                s = s_
    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.to(self.device)
        
        return torch.clip(self.actor(state), -self.action_high, 
                          self.action_high).cpu().detach().tolist()
    
    def evaluate(self, state):
        return self.critic(state, self.act(state))
    
    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(),
                                       self.target_critic.parameters()):
         target_param.data.copy_(self.tau * param.data + \
                                 (1 - self.tau) * target_param.data)

         for param, target_param in zip(self.actor.parameters(), 
                        self.target_actor.parameters()):
             target_param.data.copy_(self.tau * param.data + 
                                (1 - self.tau) * target_param.data)
    def sample(self, batch_size = 64):
        t = random.sample(self.replay_buffer, batch_size)
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
            to(self.device).float()
        states_ = torch.from_numpy(np.array(states_)).to(self.device).float()
        
        return states, actions, states_, rewards, dones
        
    def train(self):
    
        states, actions, states_, rewards, dones = self.sample()

        l = []
        q_ = self.target_critic(states_, self.target_actor(states_)).\
        squeeze(-1).tolist()
        not_dones = (1-dones).tolist()
        for i in range(len(q_)):
            l.append(q_[i]*not_dones[i])
        l = np.array(l)
        l = torch.from_numpy(l).float().to(self.device)
        with torch.no_grad(): 
           target_q = rewards + self.gamma * l

        q = self.critic(states, actions)
        critic_loss = self.loss(target_q.unsqueeze(-1),q)

        self.critic.optim.zero_grad()
        critic_loss.backward()
        self.critic.optim.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor.optim.zero_grad()
        actor_loss.backward()
        self.actor.optim.step()
        self.soft_update()
        
        
