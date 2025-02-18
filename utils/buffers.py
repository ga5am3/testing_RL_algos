import torch
import numpy as np
#from abc import ABC, abstractmethod
from collections import deque, namedtuple
import random

class SimpleBuffer():
    """
    Simple Buffer to store experiences (state, action, reward, next_state, terminated, truncated)
    """
    def __init__(self, max_size, batch_size, gamma, n_steps=1, seed=0):
        self.memory = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_steps
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated", "truncated"])
        random.seed(seed)

    def add(self, state, next_state, action, reward, terminated, truncated):
        # TODO: add the next_state to the buffer (remember to add real_next_state if truncated)?
        done = terminated or truncated

        self.n_step_buffer.append((state, action, reward, next_state, terminated, truncated))

        # when we accumulated n step transitions, we add the n-step return to the memory
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, terminated, truncated = self._calc_n_step_return(self.n_step_buffer)
            self.memory.append(self.experience(state, action, reward, next_state, terminated, truncated))
        
        # if the episode is done, we add the remaining transitions to the memory
        if done:
            while len(self.n_step_buffer):
                state, action, reward, next_state, terminated, truncated = self._calc_n_step_return(self.n_step_buffer)
                self.memory.append(self.experience(state, action, reward, next_state, terminated, truncated))
                self.n_step_buffer.popleft()        

    def _calc_n_step_return(self, buffer):
        R = 0.0

        for idx, transition in enumerate(buffer):
            _, _, reward, _, terminated, truncated = transition
            R += reward * (self.gamma ** idx)
            if terminated or truncated:
                break
        
        s = buffer[0][0]
        a = buffer[0][1]
        # next state is from the last accumulated transition
        ns = buffer[idx][3]
        te = buffer[idx][4]
        tr = buffer[idx][5]
        return s, a, R, ns, te, tr


    def sample(self, batch_size):
        if batch_size is None:
            batch_size = self.batch_size
            
        samples = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, terminals, truncated = zip(*samples)
        return states, actions, rewards, next_states, terminals, truncated

    def __len__(self):
        return len(self.memory)
    
class PER_Buffer(SimpleBuffer):
    #TODO Implement the Prioritized Experience Replay Buffer
    def __init__(self, max_size, batch_size, gamma, n_steps=1, alpha=0.6, beta_start=0.4, beta_frames=100_000, seed=0):
        super(PER_Buffer, self).__init__(max_size, batch_size, gamma, n_steps, seed)
        
        self.alpha = alpha
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        # priorities
        self.priorities = np.zeros(max_size)
        self.memory = np.zeros(max_size, dtype=np.float32)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated", "truncated"])
        
        self.pos = 0
        self.size = 0
        np.random.seed(seed)

    def beta_by_frame(self):
        """ Linearly increase beta from beta_start to 1 over beta_frames """
        return min(1.0, self.beta + self.frame * (1.0 - self.beta) / self.beta_frames)

    def _store(self, experience):
        """ Store experience and priority """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.memory[self.pos] = experience
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    def add(self, state, next_state, action, reward, terminated, truncated):
        done = terminated or truncated
        
        self.n_step_buffer.append((state, action, reward, next_state, terminated, truncated))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, terminated, truncated = self._calc_n_step_return(self.n_step_buffer)
            self._store(self.experience(state, action, reward, next_state, terminated, truncated))
        if done:
            while self.n_step_buffer:
                exp = self._calc_n_step_return(self.n_step_buffer)
                self._store(exp)
                self.n_step_buffer.popleft()

    def sample(self, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
        
        N = self.size
        if N == 0:
            return None
        
        prios = self.priorities[:N]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(N, batch_size, p=probs)
        samples = self.memory[indices]

        beta = self.beta_by_frame()
        self.frame += 1

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # unpack samples
        states, actions, rewards, next_states, terminals, truncated = zip(*samples)
        # states = torch.FloatTensor(np.concatenate(states)).to(self.device)
        # next_states = torch.FloatTensor(np.concatenate(next_states)).to(self.device)
        # actions = torch.cat(actions).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # terminated_flags = torch.FloatTensor(terminated_flags).unsqueeze(1).to(self.device)
        # truncated_flags = torch.FloatTensor(truncated_flags).unsqueeze(1).to(self.device)
        # weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, terminals, truncated, weights, indices

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-5

    def __len__(self):
        return self.size
