import torch
from torch import nn
from abc import ABC, abstractmethod
import gymnasium as gym
from utils.utils import BatchRenorm
class BaseActor(nn.Module, ABC):
    def __init__(self):
        super(BaseActor, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        foward pass of the actor, outputs mean and log_std of the action distribution
        return: mean, log_std
        """
        pass

    @abstractmethod
    def get_action(self, x):
        """
        return: action, log_prob, mean
        """
        # Forward pass

        # Reparametrization trick

        # Enforcing action bounds

        pass

class CrossQ_SAC_Actor(BaseActor):
    def __init__(self, 
                state_dim: int,
                action_dim: int,
                env: gym.Env,
                hidden_sizes: list[int]=[256, 256],
                log_std_bounds: list[float] =[-20.0, 2.0]):
        super().__init__()
        momentum = 0.1
        self.log_min, self.log_max = log_std_bounds

        self.actor_net = nn.Sequential(
            BatchRenorm(state_dim, momentum=momentum),
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[1], momentum=momentum),
        )

        self.mean = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[1], action_dim)

        self._initialize_weights()
        # TODO: check this part (single action space is not defined in not vectorized envs)
        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0))

    def _initialize_weights(self):
        for layer in list(self.actor_net) + [self.mean, self.log_std]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.log_min + 0.5 * (self.log_max - self.log_min) * (log_std + 1)

        return mean, log_std
    
    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # Reparametrization trick
        normal = torch.distributions.Normal(mean, std)
        epsilon = normal.rsample()
        squashed_epsilon = torch.tanh(epsilon)
        
        # Action bounds
        action = self.action_scale * squashed_epsilon + self.action_bias

        # Adjust log probability to compensate for the tanh squashing.
        # Using the change-of-variable formula:
        # p_y(y) = p_x(x) * |dx/dy| => log p_y(y) = log p_x(x) + log |dx/dy|
        # log p_y(y) = log p_x(x) - sum(log(1 - tanh(x)^2))
        log_prob = normal.log_prob(epsilon) - torch.log(self.action_scale * (1 - squashed_epsilon.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean
    
# sorry gabriel, deleted it on accident
class Deterministic_Actor(BaseActor):
    def __init__(self,
                state_dim: int,
                action_dim: int,
                env: gym.Env,
                hidden_sizes: list[int]=[256, 256]):
        super().__init__()

        momentum = 0.1  
        self.actor_net = nn.Sequential(
            BatchRenorm(state_dim, momentum=momentum),
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            BatchRenorm(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], action_dim)
        )

        self._initialize_weights()

        self.register_buffer("action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0))

    def _initialize_weights(self):
        for layer in list(self.actor_net) + [self.mean]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.actor_net(state)
        return x * self.action_scale + self.action_bias

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)