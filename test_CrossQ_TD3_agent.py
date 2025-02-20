import numpy as np
import gymnasium as gym
from utils.buffers import SimpleBuffer
import torch
from algos.CrossQ import CrossQTD3_Agent

if __name__ == "__main__":
    # Dummy replay buffer for testing purposes
    env = gym.make('Pendulum-v1')
    batch_size = 2
    
    print(env.reset())

    buffer = SimpleBuffer(max_size=1500, batch_size=32, gamma=0.99, n_steps=10, seed=0)
    #do_random_actions(env, buffer, max_action, batch_size)
    print("Stored transitions:", len(buffer))

    agent = CrossQTD3_Agent(env, replay_buffer=buffer)
    agent.save('/home/gabrielga/Gabo/barn/testing_RL_algos/CrossQTD3_Agent.pth')
    agent.load('/home/gabrielga/Gabo/barn/testing_RL_algos/CrossQTD3_Agent.pth')