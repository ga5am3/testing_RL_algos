from algos.CrossQ import CrossQSAC_Agent, CrossQTD3_Agent
from utils.buffers import SimpleBuffer
import gymnasium as gym
import wandb

if "__main__" == __name__:

    replay_buffer = SimpleBuffer(max_size=int(10e6), batch_size=256, gamma=0.99, n_steps=1, seed=0)

    env = gym.make("Pendulum-v1")
    agent = CrossQSAC_Agent(env, replay_buffer=replay_buffer, use_wandb=True)
    batch_size = 256
    rollout_eps = 1
    total_steps = 20000
    save_freq = 2000000
    
    agent.train(batch_size, rollout_eps, total_steps, save_freq)