from torch import nn
from abc import ABC, abstractmethod

class BaseActor(nn.Module, ABC):
    def __init__(self):
        super(BaseActor, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_action(self, x):
        pass

    @abstractmethod
    def get_log_prob(self, x, action):
        pass

    @abstractmethod
    def get_entropy(self, x):
        pass
    
    @abstractmethod
    def weight_init(self, mean=0, std=1.0):
        pass

class CrossQ_SAC_Actor(BaseActor):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass
