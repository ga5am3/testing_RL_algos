import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from utils.utils import BatchRenorm

class BaseCritic(nn.Module, ABC):
    def __init__(self):
        super(BaseCritic, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

class TwinQCritic(BaseCritic):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )
        self._initialize_weights()


    def _initialize_weights(self):
        for layer in list(self.q1) + list(self.q2):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


def get_activation(activation_choice: str) -> nn.Module:
    if activation_choice.lower() == "relu6":
        return nn.ReLU6
    elif activation_choice.lower() == "tanh":
        return nn.Tanh
    elif activation_choice.lower() == "elu":
        return nn.ELU
    elif activation_choice.lower() == "relu":
        return nn.ReLU
    else:
        raise ValueError(f"Unsupported activation function: {activation_choice}")

# Example usage:
class CrossQCritic(TwinQCritic):
    def __init__(self, state_dim, action_dim, hidden_sizes=[512, 512], activation="tanh"):
        super().__init__(state_dim, action_dim, hidden_sizes)
        self.activation = get_activation(activation)
        momentum = 0.01

        self.q1 = nn.Sequential(
            BatchRenorm(state_dim + action_dim, momentum=momentum),
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            self.activation(),
            BatchRenorm(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            self.activation(),
            BatchRenorm(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1)
        )
        self.q2 = nn.Sequential(
            BatchRenorm(state_dim + action_dim, momentum=momentum),
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            self.activation(),
            BatchRenorm(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            self.activation(),
            BatchRenorm(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1)
        )
        self._initialize_weights()


    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)