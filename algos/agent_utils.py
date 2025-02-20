import torch
import numpy as np

class Base_Agent:
    def __init__(self, env, replay_buffer, max_action):
        self.env = env
        self.replay_buffer = replay_buffer
        self.max_action = max_action
        
    def _do_random_actions(self, batch_size: int) -> None:
        # Sample random actions depending on the type of action space
        actions = np.array([self.env.action_space.sample() for _ in range(batch_size)])

        for i in range(batch_size):
            state = self.env.reset()
            terminated, truncated = False, False  
            while not terminated and not truncated:  
                action = actions[i]  # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                state = next_state
                
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    