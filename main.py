from algos.CrossQ import CrossQSAC_Agent, CrossQTD3_Agent

import gymnasium as gym
import wandb

if "__main__" == __name__:

    env = gym.make("CartPole-v0")
    agent = CrossQSAC_Agent(env,use_wandb=True, use_tensorboard=True)
    agent.train()