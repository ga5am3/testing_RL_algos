import torch
import numpy as np

class Base_Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        
    def _do_random_actions(self, initial_steps: int) -> None:
        # Sample random actions depending on the type of action space
        actions = np.array([self.env.action_space.sample() for _ in range(initial_steps)])

        for i in range(initial_steps):
            state, _ = self.env.reset()
            terminated, truncated = False, False  
            while not terminated and not truncated:  
                action = actions[i]  # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = self.env.step(action)
                print(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, terminated: {terminated}, truncated: {truncated}, info: {info}")
                self.replay_buffer.add(state, action, reward, next_state, terminated, truncated)
                state = next_state
                
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    