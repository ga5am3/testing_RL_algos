import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SAC_Agent:
    def __init__(self):
        pass

    def select_action(self, states: torch.Tensor) -> torch.Tensor:
        pass

    def update(self, replay_buffer):
        pass

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass
    

class TD3_Agent:
    def __init__(self):
        pass

    def select_action(self, states: torch.Tensor) -> torch.Tensor:
        pass

    def update(self, replay_buffer):
        pass

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass
