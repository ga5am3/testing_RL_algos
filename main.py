from algos.CrossQ import CrossQSAC_Agent, CrossQTD3_Agent
from utils.buffers import SimpleBuffer
import gymnasium as gym
import wandb

if "__main__" == __name__:

    replay_buffer = SimpleBuffer(max_size=1000, batch_size=50, gamma=0.99, n_steps=1, seed=0)

    env = gym.make("Pendulum-v1")
    agent = CrossQSAC_Agent(env, replay_buffer=replay_buffer, use_wandb=True)
    batch_size = 50
    rollout_eps = 10
    total_steps = 1000
    save_freq = 500
    agent.train(batch_size, rollout_eps, total_steps, save_freq)