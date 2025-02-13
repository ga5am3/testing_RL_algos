import torch
import torch.nn as nn

from abc import ABC, abstractmethod

# import sys
# sys.path.append('/home/bbruno/Documents/testing_RL_algos/')
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
        # TODO: Implement Orthogonal Initialization
        for layer in list(self.q1) + list(self.q2):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

# class TwinQCritic_separate_prepross(BaseCritic):
#     def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.state_net_q1 = nn.Sequential(
#             nn.Linear(state_dim, hidden_sizes[0]),
#             nn.ReLU(),
#         )
#         self.state_net_q2 = nn.Sequential(
#             nn.Linear(state_dim, hidden_sizes[0]),
#             nn.ReLU(),
#         )
#         self.action_net_q1 = nn.Sequential(
#             nn.Linear(action_dim, hidden_sizes[0]),
#             nn.ReLU(),
#         )
#         self.action_net_q2 = nn.Sequential(
#             nn.Linear(action_dim, hidden_sizes[0]),
#             nn.ReLU(),
#         )
#         self.shared_q1 = nn.Sequential(
#             nn.Linear(hidden_sizes[0]*2, hidden_sizes[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_sizes[1], 1)
#         )
#         self.shared_q2 = nn.Sequential(
#             nn.Linear(hidden_sizes[0]*2, hidden_sizes[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_sizes[1], 1)
#         )
#         self._initialize_weights()


#     def _initialize_weights(self):
#         for layer in list(self.q1) + list(self.q2):
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)

#     def forward(self, state, action):
#         state_x1 = self.state_net_q1(state)
#         state_x2 = self.state_net_q2(state)
#         action_x1 = self.action_net_q1(action)
#         action_x2 = self.action_net_q2(action)
#         x1 = torch.cat([state_x1, action_x1], dim=1)
#         x2 = torch.cat([state_x2, action_x2], dim=1)
#         x1 = self.shared_q1(x1)
#         x2 = self.shared_q2(x2)
#         return x1, x2

class CrossQCritic(TwinQCritic):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__(state_dim, action_dim, hidden_sizes)

        momentum = 0.01
        # ?? Idk if I should put the output size to be 2 or 1
        # ?? I think it should be 2 because we are trying to predict the Q value for this time step and the next time step
        # ?? And idk if I should use state_dim + action_dim or 2*state_dim + 2*action_dim or let whoever uses this class to decide 
        self.q1 = nn.Sequential(
            BatchRenorm(state_dim + action_dim, momentum=momentum),
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1)
        )
        self.q2 = nn.Sequential(
            BatchRenorm(state_dim + action_dim, momentum=momentum),
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1)
        )
        self._initialize_weights()


    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)