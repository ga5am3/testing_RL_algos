import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from networks.actors import CrossQ_SAC_Actor, Deterministic_Actor
from networks.critics import CrossQCritic
from utils.buffers import SimpleBuffer
import copy 
import gymnasium as gym

# ------| Stuffs to do |-------
# - Test _do_random_actions
# - Test rollout
# - Test sampling from the buffer
# - Add Tensorboard logging and wandb logging
# - test saving and loading
# - verify both methods are interchangable
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
        self.initial_training_steps = 1000 #TODO verify if these are steps or episodes
        self.training_steps_per_rollout = 1 
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.device = device if device is not None else get_device()

        self.replay_buffer = None if replay_buffer is None else replay_buffer
        # initialize the actor and critic networks
        self.actor = CrossQ_SAC_Actor(state_dim, 
                                      action_dim, 
                                      env=env,
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
        self.log_alpha = torch.tensor([np.log(init_temperature)], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr, betas=(0.5, 0.999))

    #TODO: check if this is necessary
    def select_action(self, states: torch.Tensor, train: bool) -> torch.Tensor:
        """
        input: state (torch.Tensor)
        output: action (torch.Tensor)
        """
        # get the action from the actor (no gradients)
        self.actor.eval()
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = states.unsqueeze(0)
            if train:
                action, _, _ = self.actor.get_action(states)
            else:
                _ , _, action = self.actor.get_action(states)
            action = action.cpu().numpy().flatten()
        self.actor.train()
        return action

    def rollout(self, env, episodes: int, eval: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # Run policy in environment
        for _ in range(episodes):
            print("====================================")
            print("Rollout step ", _)
            state, _ = env.reset(seed=0) #TODO: make seed a parameter
            termination = False
            truncation = False
            # steps = 0
            while ((not termination) and (not truncation)):
                action = self.select_action(state, eval).cpu().numpy()
                next_state, reward, termination, truncation, infos = env.step(action)                
                self.replay_buffer.add(state, next_state, action, reward, termination, truncation)
                state = next_state  
                print("State: ", state) 
                print("Termination: ", termination)
                print("Truncation: ", truncation)
                # steps += 1

    def train(self, batch_size: int, total_timesteps: int, save_freq: int = 1000) -> None:
        """
        Train the agent
        """

        # TODO: check how much to fill the replay buffer.
        if len(self.replay_buffer) == 0:
            self._do_random_actions(batch_size)

        for global_step in range(total_timesteps):
            self.rollout(self.env, self.initial_training_steps, eval=False) 
            
            # rollout the agent in the environment
            # sample a batch from the replay buffer
            for train_step in range(self.training_steps_per_rollout):
                experience = self.replay_buffer.sample(batch_size)
                states, next_states, actions, rewards, terminations, truncations = experience
                states = torch.FloatTensor(states).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                terminations = torch.FloatTensor(terminations).to(self.device)
                truncations = torch.FloatTensor(truncations).to(self.device)

                # calculate the Q
                with torch.no_grad():
                    self.actor.eval()
                    next_actions, log_probs, _ = self.actor.get_action(next_states)
                    self.actor.train()

                cat_states = torch.cat([states, next_states], dim=0)
                cat_actions = torch.cat([actions, next_actions], dim=0)
                cat_q1, cat_q2 = self.critic(cat_states, cat_actions)

                q_values_1, q_values_2 = torch.chunk(cat_q1, chunks=2, dim=0)
                q_values_1_next, q_values_2_next = torch.chunk(cat_q2, chunks=2, dim=0)

                target_q_values = (torch.min(q_values_1_next, q_values_2_next) - self.alpha * log_probs)
                
                q_target = rewards * self.rewards_scale + self.gamma * (1 - terminations) * target_q_values
                torch.detach(q_target)

                q1_loss = F.mse_loss(q_values_1, q_target)
                q2_loss = F.mse_loss(q_values_2, q_target)
                total_q_loss = q1_loss + q2_loss
                
                self.critic_net_optimizer.zero_grad()
                total_q_loss.backward()
                self.critic_net_optimizer.step()
                # log actor loss, critic loss, entropy loss, alpha

                if global_step % self.policy_update_freq == 0:
                    # policy update
                    next_actions, log_probs, _ = self.actor.get_action(states)
                    
                    self.critic.eval()
                    q1, q2 = self.critic(states, next_actions)
                    self.critic.train()

                    min_q = torch.min(q1, q2)
                    policy_loss = (self.alpha * log_probs - min_q).mean()

                    self.actor_net_optimizer.zero_grad()
                    policy_loss.backward()
                    self.actor_net_optimizer.step()

                    # temperature update
                    entropy_loss = (self.alpha * (-log_probs - self.target_entropy).detach()).mean()
                    self.alpha_optimizer.zero_grad()
                    entropy_loss.backward()
                    self.alpha_optimizer.step()

                    # log actor loss, entropy loss

                # Save the model checkpoint every save_freq training steps
                if global_step % save_freq == 0 and global_step > 0:
                    self.save(f"model_checkpoint_{global_step}.pt")

    def _do_random_actions(self, batch_size: int) -> None:
        # Sample random actions depending on the type of action space
        actions = np.array([self.env.action_space.sample() for _ in range(batch_size)])

        for i in range(batch_size):
            state = self.env.reset()
            terminated, truncated = False, False  
            while not terminated and not truncated:  
                action = actions[i]  # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                state = next_state

    def save(self, filename: str) -> None:
        """
        Save the models in the agent
        """
        #TODO: Check if this is the correct way to save the models or whether to save actor and critic separately
        state = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_net_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_net_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'max_action': self.max_action,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_update_freq': self.policy_update_freq,
        }
        torch.save(state, filename)

    def load(self, filename: str) -> None:
        """
        Load the models in the agent
        """
        state = torch.load(filename)
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.actor_net_optimizer.load_state_dict(state['actor_optimizer_state_dict'])
        self.critic_net_optimizer.load_state_dict(state['critic_optimizer_state_dict'])
        self.log_alpha = state['log_alpha']
        self.alpha_optimizer.load_state_dict(state['alpha_optimizer_state_dict'])
        self.max_action = state['max_action']
        self.gamma = state['gamma']
        self.tau = state['tau']
        self.policy_update_freq = state['policy_update_freq']
    
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
                noise = torch.normal(0, self.max_action * self.exploration_noise, size=self.action_dim) # ? Do we get the max action from env?
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

    def rollout(self, env, max_steps: int, train: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # Run policy in environment
        for _ in range(max_steps):
            state = env.reset()
            terminated = False
            truncated = False
            while not terminated or not truncated: #TODO: add behavior for truncated episodes
                action = self.select_action(state, train) # While rollouts the target actor is used
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
            
            self.rollout(env, max_steps, train=True)

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
                    target_q = torch.minimum(q_next_1, q_next_2)
                    done = terminated_tensor if terminated_tensor.item() == 1 else torch.tensor(0, dtype=torch.float32)
                    target_q = reward_tensor + (1 - done) * self.gamma * target_q

                    
        # calculate and update critic loss

        # log stuff

        # every N steps update the actor 

        # update target networks
        # update info

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass

