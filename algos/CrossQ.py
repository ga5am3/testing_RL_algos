import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.actors import CrossQ_SAC_Actor, Deterministic_Actor
from networks.critics import CrossQCritic
from utils.buffers import SimpleBuffer
import copy 
import gymnasium as gym

class CrossQSAC_Agent:
    """
    Original Paper: https://arxiv.org/abs/1801.01290
    """
    def __init__(self, env,
                 actor_hidden_layers: list[int]=[256, 256],
                 critic_hidden_layers: list[int]=[256, 256],
                 actor_lr: float=3e-4, critic_lr: float = 3e-4, 
                 max_action: float=1.0, device: str=None,
                 gamma=0.99, tau=5e-3, policy_freq=2, 
                 replay_buffer=None):
        
        self.env = env
        self.learning_steps = 0
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.device = device if device is not None else get_device()

        self.replay_buffer = None if replay_buffer is None else replay_buffer
        # initialize the actor and critic networks
        self.actor = CrossQ_SAC_Actor(state_dim, 
                                      action_dim, 
                                      hidden_sizes=actor_hidden_layers).to(self.device)
        self.critic = CrossQCritic(state_dim=state_dim, 
                                   action_dim=action_dim, 
                                   hidden_sizes=critic_hidden_layers, 
                                   activation="tanh").to(self.device)
        # define optimizers
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.rewards_scale = 1.0
        self.policy_update_freq = policy_freq

        self.actor_net_optimizer = optim.Adam(self.actor.parameters(),
                                                lr=actor_lr, betas=(0.5, 0.999))
        self.critic_net_optimizer = optim.Adam(self.critic.parameters(),
                                                lr=critic_lr, betas=(0.5, 0.999))
        # entropy tunner
        self.target_entropy = -torch.prod(torch.tensor(env.action_space.shape).to(self.device)) #TODO: check this 
        init_temperature = 0.1
        self.log_alpha = torch.tensor([np.log(init_temperature)], 
                                      requires_grad=True).to(self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr, betas=(0.5, 0.999))

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

    def train(self, batch_size: int, total_timesteps: int) -> None:
        """
        Train the agent
        """
        # TODO: check how much to fill the replay buffer.
        if len(self.replay_buffer) == 0:
            self._do_random_actions(batch_size)

        for global_step in range(total_timesteps):
            self.rollout(self.env, max_steps, eval=False) #TODO fix max_steps

            # rollout the agent in the environment
            # sample a batch from the replay buffer
            experience = self.replay_buffer.sample(batch_size)
            states, next_states, actions, rewards, dones = experience

            batch_size = len(states)

            # convert everything to tensors
            # calculate the Q
            # calculate and update critic loss

            # log stuff

            # every N steps update the actor and entropy tuner

            # update target networks
            # update info
        pass

    def _do_random_actions(self, batch_size: int) -> None:
        actions = np.random.uniform(
            -self.max_action,
            self.max_action,
            size=(batch_size, self.env.action_space.shape[0])
        )
        next_states, rewards, terminations, truncations, infos = self.env.step(actions)
        real_next_obs = next_states
        self.replay_buffer.add_batch(states, actions, rewards,
                                     real_next_obs, terminations, truncations)

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
    
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class TD3_Agent:
    """
    Original Paper: https://arxiv.org/abs/1802.09477v3
    """
    def __init__(self, env: gym.Env, 
                 state_dim: int, 
                 action_dim: int, 
                 actor_hidden_layers: list[int], 
                 critic_hidden_layers: list[int], 
                 actor_lr: float, 
                 critic_lr: float, 
                 max_action, 
                 device: str = None,
                 gamma: float = 0.99, 
                 tau: float = 0.005, 
                 policy_noise: float = 0.2, 
                 noise_clip=0.5, 
                 exploration_noise: float = 0.1,
                 policy_freq=2):
        
        self.device = device if device is not None else get_device()
        self.action_dim = action_dim
        
        # initialize the actor and critic networks
        self.actor = Deterministic_Actor(state_dim, action_dim, env, actor_hidden_layers).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = CrossQCritic(state_dim, action_dim, env, critic_hidden_layers, activation="tanh").to(self.device)
        
        # Actor target netowrk is always used for evaluation only
        self.target_actor.eval()
        
        # define parameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        #self.policy_freq = policy_freq
        #self.total_it = 0
        
        self.replay_buffer = None # TODO find a way to set a simple buffer or PER buffer
        
        # define optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)


    def select_action(self, state: torch.Tensor, train: bool) -> torch.Tensor:
        """
        input: states (torch.Tensor)
        output: actions (torch.Tensor)
        """
        if train:
            self.actor.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device) # ? should I use torch.FloatTensor or just use the torch.Tensor type for state. 
                action = self.actor(state).cpu().data.numpy().flatten()
                noise = np.random.normal(0, self.max_action * self.exploration_noise, size=self.action_dim) # ? Do we get the max action from env?
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
                action = torch.Tensor(action)
            self.actor.train()
            
        else:
            self.target_actor.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action = self.target_actor(state).cpu().data.numpy().flatten()
                noise = torch.normal(0, self.policy_noise, size=self.action_dim).clamp(-self.noise_clip, self.noise_clip)
                action = action + noise
                action = action.clamp(-self.max_action, self.max_action)
            self.target_actor.train()
        
        return action

    def rollout(self, env, max_steps: int, eval: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # Run policy in environment
        for _ in range(max_steps):
            state = env.reset()
            while not terminated or not truncated:
                action = self.select_action(state, False) # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = env.step(action)
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                state = next_state

    def train(self, env, max_steps, max_size, gamma, batch_size: int, train_episodes, train_steps) -> None:
        """
        Train the agent
        """
        # rollout the agent in the environment
        #TODO define a way to set the buffer
        self.replay_buffer = SimpleBuffer(max_size, batch_size, gamma, n_steps=1, seed=0)
        self.rollout(env, max_steps, eval=False)
        
        ep_counter = 0
        
        for ep in range(train_episodes):
            state = env.reset()
            
            while not terminated or not truncated:
                action = self.select_action(state, train=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                
                # Train 50 times every 50 steps
                if ep_counter % train_steps == 0:                
                    # sample a batch from the replay buffer
                    state, next_state, action, reward, terminated, truncated = self.replay_buffer.sample(batch_size)
            
                    # convert everything to tensors
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                    action_tensor = torch.FloatTensor(action).to(self.device)
                    reward_tensor = torch.FloatTensor(reward).to(self.device)    
                    terminated_tensor = torch.tensor(terminated, dtype=torch.float32, device=self.device) # ? Are terminated and truncated boolean values?   
                    truncated_tensor = torch.tensor(truncated, dtype=torch.float32, device=self.device) # ? do we need truncated values?
        
                    # calculate the Q
                    self.critic.eval()
                    with torch.no_grad():
                        #noise = (torch.randn_like(action_tensor) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                        next_action = self.select_action(next_state, train=False)
                        state_next_state = torch.cat([state_tensor, next_state_tensor], dim=0) # * If state tensor is 1D, put an unsqueeze(0)
                        action_next_action = torch.cat([action_tensor, next_action], dim=0) # * If state tensor is 1D, put an unsqueeze(0)
                        q_vals_1, q_vals_2 = self.critic(state_next_state, action_next_action)
                        q_curr_1, q_next_1 = torch.chunk(q_vals_1, chunks=2, dim=0)
                        q_curr_2, q_next_2 = torch.chunk(q_vals_2, chunks=2, dim=0)
                        target_q = torch.min(q_next_1, q_next_2)
                        done = terminated_tensor if terminated_tensor.item() == 1 else torch.tensor(0, dtype=torch.float32)
                        target_q = reward_tensor + (1 - done) * self.gamma * target_q
                        
                        
                        
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward_tensor + (1 - terminated_tensor) * self.gamma * target_Q

                    current_Q1, current_Q2 = self.critic(state_tensor, action_tensor)
                    
        # calculate and update critic loss

        # log stuff

        # every N steps update the actor 

        # update target networks
        # update info

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass

