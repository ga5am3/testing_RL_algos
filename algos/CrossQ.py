import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..networks.actors import CrossQ_SAC_Actor
import copy 

class CrossQSAC_Agent:
    """
    Original Paper: https://arxiv.org/abs/1801.01290
    """
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_sizes=[256, 256], lr=3e-4, gamma=0.99, tau=5e-3, alpha=0.2):
        # initialize the actor and critic networks
        # define parameters
        # define optimizers
        # entropy tunner
        pass

    def select_action(self, states: torch.Tensor, eval: bool) -> torch.Tensor:
        """
        input: states (torch.Tensor)
        output: actions (torch.Tensor)
        """
        # get the actions from the actor (no gradients)
        pass

    def rollout(self, env, max_steps: int, eval: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # Run policy in environment

        pass

    def train(self, batch_size: int) -> None:
        """
        Train the agent
        """
        # rollout the agent in the environment
        # sample a batch from the replay buffer

        # convert everything to tensors
        # calculate the Q
        # calculate and update critic loss

        # log stuff

        # every N steps update the actor and entropy tuner

        # update target networks
        # update info
        pass

    def save(self, filename: str) -> None:
        """
        Save the models in the agent
        """
        pass

    def load(self, filename: str) -> None:
        """
        Load the models in the agent
        """
        pass
    

class TD3_Agent:
    """
    Original Paper: https://arxiv.org/abs/1802.09477v3
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 actor_hidden_layers: list[int], critic_hidden_layers: list[int], 
                 max_action, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        # initialize the actor and critic networks
        self.actor = Actor(state_dim, action_dim, actor_hidden_layers) #max_action (optionalto scale) .to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim + action_dim, 1, critic_hidden_layers)
        
        self.target_actor.eval()
        
        # define parameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise # ???
        self.noise_clip = noise_clip # ???
        self.policy_freq = policy_freq # ???
        self.total_it = 0
        
        # define optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)


    def select_action(self, states: torch.Tensor) -> torch.Tensor:
        """
        input: states (torch.Tensor)
        output: actions (torch.Tensor)
        """
        # get the actions from the actor (no gradients)
        self.actor.eval()
        with torch.no_grad():
            states = torch.FloatTensor(states).to(device)
            action = self.actor(states).cpu().data.numpy().flatten()
        
        self.actor.train()
        return action

    def rollout(self, env, max_steps: int, eval: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # Run policy in environment

        
        
        pass

    def train(self, batch_size: int) -> None:
        """
        Train the agent
        """
        # rollout the agent in the environment
        # sample a batch from the replay buffer

        # convert everything to tensors
        # calculate the Q
        # calculate and update critic loss

        # log stuff

        # every N steps update the actor 

        # update target networks
        # update info

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass
