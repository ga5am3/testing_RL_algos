import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from networks.actors import CrossQ_SAC_Actor, Deterministic_Actor
from networks.critics import CrossQCritic
from algos.agent_utils import Base_Agent
from utils.buffers import SimpleBuffer
import copy 
import gymnasium as gym

from tensorboardX import SummaryWriter
import wandb

#wandb.init(sync_tensorboard=True)
#wandb.tensorboard.patch(root_logdir="logs")

log_dir = "logs" #TODO: change this later to work with config

# ------| Stuffs to do |-------
# - Add Tensorboard logging and wandb logging
# - test saving and loading
# - verify both methods are interchangable
class CrossQSAC_Agent(Base_Agent):
    """
    Original Paper: https://arxiv.org/abs/1801.01290
    """
    def __init__(self, env: gym.Env,
                 actor_hidden_layers: list[int]=[256, 256],
                 critic_hidden_layers: list[int]=[256, 256],
                 actor_lr: float=3e-4, critic_lr: float = 3e-4, 
                 max_action: float=1.0, device: str=None,
                 gamma=0.99, tau=5e-3, policy_freq=2, 
                 replay_buffer=None, use_wandb=False):
        
        self.env = env
        self.learning_steps = 0
        self.initial_training_steps = 1000 #TODO verify if these are steps or episodes
        self.training_steps_per_rollout = 1 
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.device = device if device is not None else self.get_device()

        self.replay_buffer = None if replay_buffer is None else replay_buffer
        # initialize the actor and critic networks
        # - send high and low action values as parameters instead of the env
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

        if use_wandb: #TODO: this should be in the main file
            wandb.init(
                project="crossq_project", # set this from config file
                config={
                    "actor_hidden_layers": actor_hidden_layers,
                    "critic_hidden_layers": critic_hidden_layers,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "max_action": max_action,
                    "gamma": gamma,
                    "tau": tau,
                    "policy_freq": policy_freq,
                }
            )


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
            #print("====================================")
            print("Rollout step ", _)
            state, _ = env.reset(seed=0) #TODO: make seed a parameter
            termination = False
            truncation = False

            total_ep_reward = 0
            steps = 0
            while ((not termination) and (not truncation)):
                action = self.select_action(state, eval).cpu().numpy()
                next_state, reward, termination, truncation, infos = env.step(action)                
                self.replay_buffer.add(state, next_state, action, reward, termination, truncation)
                state = next_state 
                steps += 1
                total_ep_reward += reward

            print(f"Episode finished in {steps} steps with Average Reward = {total_ep_reward:.2f}")
            if self.use_wandb:
                wandb.log({
                    "rollout/episode_steps": steps,
                    "rollout/episode_reward": total_ep_reward
                })

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
                
                # Compute gradient and do optimizer step logging #! might remove this
                critic_grad_norm = sum([p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters()]) ** 0.5

                if self.use_wandb:
                    wandb.log({
                        "critic_1_loss": q1_loss,
                        "critic_2_loss": q2_loss,
                        "critic_loss": total_q_loss,
                        "entropy_loss": entropy_loss,
                        "critic_grad_norm": critic_grad_norm,
                        "train_step": global_step
                    })

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
                    if self.use_wandb:
                        wandb.log({
                            "actor_loss": policy_loss,
                            "entropy_loss": entropy_loss,
                            "alpha": self.alpha,
                            "log_probs": log_probs,
                        })

                # Save the model checkpoint every save_freq training steps
                if global_step % save_freq == 0 and global_step > 0:
                    self.save(f"model_checkpoint_{global_step}.pt")

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

class CrossQTD3_Agent(Base_Agent):
    """
    Original Paper: https://arxiv.org/abs/1802.09477v3
    """
    def __init__(self, env: gym.Env, 
                 actor_hidden_layers: list[int] = [256, 256], 
                 critic_hidden_layers: list[int] = [256, 256], 
                 actor_lr: float = 3e-4, 
                 critic_lr: float = 3e-4, 
                 device: str = None,
                 gamma: float = 0.99, 
                 tau: float = 0.005, 
                 policy_noise: float = 0.2, 
                 noise_clip: float = 0.5, 
                 exploration_noise: float = 0.1,
                 policy_freq_update: int = 2, 
                 replay_buffer=None):
        
        self.device = device if device is not None else self.get_device()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # initialize the actor and critic networks
        self.actor = Deterministic_Actor(state_dim, action_dim, env, actor_hidden_layers).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = CrossQCritic(state_dim, action_dim, critic_hidden_layers, activation="tanh").to(self.device)
        
        # Actor target netowrk is always used for evaluation only
        self.target_actor.eval()
        
        # define parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_freq_update = policy_freq_update
        
        self.replay_buffer = None if replay_buffer is None else replay_buffer
        
        # define optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.5, 0.999))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(0.5, 0.999))


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

    def rollout(self, max_steps: int, train: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # Run policy in environment
        for _ in range(max_steps):
            state = self.env.reset()
            terminated = False
            truncated = False

            total_ep_reward = 0
            steps = 0
            while not terminated and not truncated: #TODO: add behavior for truncated episodes
                action = self.select_action(state, train) # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                state = next_state

                # update episode statistics
                steps += 1
                total_ep_reward += reward
    
            print(f"Episode finished in {steps} steps with Average Reward = {total_ep_reward:.2f}")
            if self.use_wandb:
                wandb.log({
                    "rollout/episode_steps": steps,
                    "rollout/episode_reward": total_ep_reward
                })

    def train(self, env, max_steps, max_size, gamma, batch_size: int, train_episodes, train_steps_per_rollout) -> None:
        """
        Train the agent
        """
        
        if len(self.replay_buffer) == 0:
            self._do_random_actions(batch_size)
        
        for global_step in range(train_episodes):
            
            self.rollout(max_steps, train=True)
            
            for _ in range(train_steps_per_rollout):

                # Train 50 times every 50 steps
                #if ep_counter % train_steps == 0:                
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
                torch.detach(target_q)

                critic_loss_1 = F.mse_loss(q_curr_1, target_q)
                critic_loss_2 = F.mse_loss(q_curr_2, target_q)

                total_critic_loss = critic_loss_1 + critic_loss_2

                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                self.critic_optimizer.step()

                # Compute gradient and do optimizer step logging #! might remove this
                critic_grad_norm = sum([p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters()]) ** 0.5

                if self.wandb:
                    wandb.log({
                        "critic_1_loss": critic_loss_1,
                        "critic_2_loss": critic_loss_2,
                        "critic_total_loss": total_critic_loss,
                        "critic_grad_norm": critic_grad_norm,
                        "train_step": global_step
                    })
                
                # every N steps update the actor
                if global_step % self.policy_freq_update == 0:
                    n_action = self.select_action(state_tensor, train=False)
                    self.critic.eval()
                    with torch.no_grad():
                        q_values_1, q_values_2 = self.critic(state_tensor, n_action)
                    min_q_value = torch.minimum(q_values_1, q_values_2)
                    actor_loss = -min_q_value.mean()
                    self.critic.train()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()       
                    self.actor_optimizer.step()
                    
                    # update target networks
                    for actor_param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.tau * actor_param.data + (1 - self.tau) * target_param.data)

                    # log stuff
                    if self.wandb:
                        wandb.log({
                            "actor_loss": actor_loss,
                        })

            
                # Save the model checkpoint every save_freq training steps
                if global_step % save_freq == 0 and global_step > 0:
                    self.save(f"model_checkpoint_{global_step}.pt")

    def save(self, filename: str) -> None:
        models = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'exploration_noise': self.exploration_noise,
            'policy_update_freq': self.policy_freq_update,
        }
        torch.save(models, filename)

    def load(self, filename: str) -> None:
        models = torch.load(filename)
        self.actor.load_state_dict(models['actor_state_dict'])
        self.critic.load_state_dict(models['critic_state_dict'])
        self.actor_optimizer.load_state_dict(models['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(models['critic_optimizer_state_dict'])
        self.gamma = models['gamma']
        self.tau = models['tau']
        self.policy_noise = models['policy_noise']
        self.noise_clip = models['noise_clip']
        self.exploration_noise = models['exploration_noise']
        self.policy_freq_update = models['policy_update_freq']
        

