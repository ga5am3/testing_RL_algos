import torch
import numpy as np
#from abc import ABC, abstractmethod
from collections import deque, namedtuple
import random

class SimpleBuffer():
    # TODO: make it work with n-step returns
    def __init__(self, max_size, batch_size, gamma, n_steps=1, seed=0):
        self.memory = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_steps
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated", "truncated"])
        random.seed(seed)
    def add(self, state, next_state, action, reward, terminated, truncated, done):
        # TODO: add the next_state to the buffer (remember to add real_next_state if truncated)?
        done = terminated or truncated

        self.n_step_buffer.append((state, action, reward, next_state, terminated, truncated))

        # when we accumulated n step transitions, we add the n-step return to the memory
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, terminated, truncated = self.calc_n_step_return(self.n_step_buffer)
            self.memory.append(self.experience(state, action, reward, next_state, terminated, truncated))

        if done:
            while len(self.n_step_buffer):
                state, action, reward, next_state, terminated, truncated = self.calc_n_step_return(len(self.n_step_buffer))
                self.memory.append(self.experience(state, action, reward, next_state, terminated, truncated))
                self.n_step_buffer.popleft()        


    def calc_n_step_return(self, buffer):
        R = 0.0

        for idx, transition in enumerate(buffer):
            _, _, reward, _, te, tr = transition
            R += reward * (self.gamma ** idx)
            if te or tr:
                break
        
        s = buffer[0][0]
        a = buffer[0][1]
        # next state is the last state in the buffer
        ns = buffer[idx][3]
        d = buffer[idx][4]
        return s, a, R, ns, d


    def sample(self, batch_size):
        if batch_size is None:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return self.size
    
class PER_Buffer(SimpleBuffer):
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

