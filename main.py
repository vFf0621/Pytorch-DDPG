import gym
import numpy as np
from gym import wrappers
from DDPG import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = DDPG(env, 192)
    ys = []
    eps = 10000
    xs = list(range(eps))
    print("Replay Buffer Initialized")
    for j in range(eps):
        done = False
        episode_reward = 0
        s = env.reset()
        while not done:
            action = agent.act(s)
            env.render()

            s_, r, done, _= env.step(action)
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
    
    

    
    

