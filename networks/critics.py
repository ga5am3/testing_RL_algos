import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class BaseCritic(nn.Module, ABC):
    def __init__(self):
        super(BaseCritic, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass
