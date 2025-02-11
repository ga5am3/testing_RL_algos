import torch
import numpy as np
from abc import ABC, abstractmethod

class ReplayBuffer(ABC):
    @abstractmethod
    def __init__(self, state_dim, action_dim, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size

    @abstractmethod
    def add(self, state, next_state, action, reward, terminated, truncated):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def __len__(self):
        pass

class SimpleBuffer(ReplayBuffer):
    # TODO: make it work with n-step returns
    def __init__(self, state_dim, action_dim, max_size):
        super().__init__(state_dim, action_dim, max_size)
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)
        self.truncated = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.size = 0
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, next_state, action, reward, terminated, truncated, done):
        # TODO: add the next_state to the buffer
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.terminated[self.ptr] = terminated
        self.truncated[self.ptr] = truncated
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # TODO: fix to add Next state
        idx = np.random.randint(0, self.size, batch_size)
        return self.state[idx], self.action[idx], self.reward[idx], self.terminated[idx], self.truncated[idx], self.done[idx]

    def __len__(self):
        return self.size
    
class PER_Buffer(ReplayBuffer):
    #TODO Implement the Prioritized Experience Replay Buffer
    def __init__(self, state_dim, action_dim, max_size):
        super().__init__(state_dim, action_dim, max_size)
        pass

    def add(self, state, action, reward, terminated, truncated, done):
        pass

    def sample(self, batch_size):
        pass

    def __len__(self):
        pass

    def update(self, idx, td_error):
        pass

    def get_priority(self, td_error):
        pass

    def get_max_priority(self):
        pass

