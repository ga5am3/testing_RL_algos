import torch
import torch.nn as nn
import torch.nn.functional as F
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

# wandb.init(sync_tensorboard=True)
# wandb.tensorboard.patch(root_logdir="logs")

log_dir = "logs"  # TODO: change this later to work with config


# ------| Stuffs to do |-------
# - Add Tensorboard logging and wandb logging
# - test saving and loading
# - verify both methods are interchangable
class CrossQSAC_Agent(Base_Agent):
    """
    Original Paper: https://arxiv.org/abs/1801.01290
    """

    def __init__(
        self,
        env: gym.Env,
        actor_hidden_layers: list[int] = [512, 512],
        critic_hidden_layers: list[int] = [512, 512],
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        device: str = None,
        gamma: float = 0.99,
        policy_freq: int = 2,
        replay_buffer: SimpleBuffer = None,
        use_wandb: bool = False,
    ):
        super().__init__(env, replay_buffer)

        self.env = env
        self.initial_training_steps = 50  # TODO verify if these are steps or episodes
        self.training_steps_per_rollout = 2
        self.use_wandb = use_wandb
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.device = device if device is not None else self.get_device()

        self.replay_buffer = None if replay_buffer is None else replay_buffer

        # initialize the actor and critic networks
        # - send high and low action values as parameters instead of the env
        self.actor = CrossQ_SAC_Actor(
            state_dim, action_dim, env=env, hidden_sizes=actor_hidden_layers
        ).to(self.device)

        self.critic = CrossQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=critic_hidden_layers,
            activation="tanh",
        ).to(self.device)
        # params
        self.max_action = env.action_space.high
        self.gamma = gamma
        self.rewards_scale = 1.0
        self.policy_update_freq = policy_freq

        # define optimizers
        self.actor_net_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(0.5, 0.999)
        )

        self.critic_net_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(0.5, 0.999)
        )

        # entropy tunner
        # TODO: check this
        self.target_entropy = -torch.prod(
            torch.tensor(env.action_space.shape).to(self.device)
        )
        init_temperature = 1.0
        self.log_alpha = torch.tensor(
            [np.log(init_temperature)],
            requires_grad=True,
            dtype=torch.float32,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=actor_lr, betas=(0.5, 0.999)
        )

        if use_wandb:  # TODO: this should be in the main file
            wandb.init(
                project="crossq_project",  # set this from config file
                config={
                    "actor_hidden_layers": actor_hidden_layers,
                    "critic_hidden_layers": critic_hidden_layers,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "max_action": self.max_action,
                    "gamma": gamma,
                    "policy_freq": policy_freq,
                },
            )

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
                _, _, action = self.actor.get_action(states)
            action = action.cpu().numpy().flatten()
        self.actor.train()
        return action

    def train(
        self, batch_size: int, rollout_eps: int, total_steps: int, save_freq: int = 1000
    ) -> None:
        """
        Train the agent
        """
        for global_step in range(total_steps):
            if len(self.replay_buffer) == 0:
                self._do_random_actions(self.initial_training_steps)

            else:
                episode_reward, steps = self.rollout(eval=True)
                if self.use_wandb:
                    wandb.log(
                        {
                            "rollout/episode_reward": episode_reward,
                            "rollout/episode_lenght": steps,
                            "rollout/avg_reward": episode_reward / steps,
                        }
                    )

            # rollout the agent in the environment
            # sample a batch from the replay buffer
            experience = self.replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, terminations, _ = (
                experience
            )
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            terminations = torch.FloatTensor(terminations).to(self.device)

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

            target_q_values = (
                torch.minimum(q_values_1_next, q_values_2_next)
                - self.log_alpha * log_probs
            )
            # print('Terminations', terminations.shape)
            # print('Rewards', rewards.shape)
            # print('Target Q values', target_q_values.shape)

            q_target = (
                rewards.unsqueeze(-1) * self.rewards_scale
                + self.gamma * (1 - terminations) @ target_q_values
            )
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
                policy_loss = (self.log_alpha * log_probs - min_q).mean()

                self.actor_net_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_net_optimizer.step()

                # temperature update
                entropy_loss = (
                    self.log_alpha * (-log_probs - self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                entropy_loss.backward()
                self.alpha_optimizer.step()

            if self.use_wandb and global_step % 10 == 0:
                # Compute gradient and do optimizer step logging #! might remove this
                critic_grad_norm = (
                    sum(
                        [
                            p.grad.data.norm(2).item() ** 2
                            for p in self.critic.parameters()
                        ]
                    )
                    ** 0.5
                )

                wandb.log(
                    {
                        "critic_1_loss": q1_loss,
                        "critic_2_loss": q2_loss,
                        "critic_loss": total_q_loss,
                        "critic_grad_norm": critic_grad_norm,
                        "global_step": global_step,
                        "actor_loss": policy_loss,
                        "entropy_loss": entropy_loss,
                        "log_alpha": self.log_alpha,
                        "log_probs": log_probs,
                    }
                )
                # Save the model checkpoint every save_freq training steps
                # if global_step % save_freq == 0 and global_step > 0:
                #    self.save(f"model_checkpoint_{global_step}.pt")

    def save(self, filename: str) -> None:
        """
        Save the models in the agent
        """
        # TODO: Check if this is the correct way to save the models or whether to save actor and critic separately
        state = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_net_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_net_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "gamma": self.gamma,
            "policy_update_freq": self.policy_update_freq,
        }
        torch.save(state, filename)

    def load(self, filename: str) -> None:
        """
        Load the models in the agent
        """
        state = torch.load(filename)
        self.actor.load_state_dict(state["actor_state_dict"])
        self.critic.load_state_dict(state["critic_state_dict"])
        self.actor_net_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        self.critic_net_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        self.log_alpha = state["log_alpha"]
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer_state_dict"])
        self.max_action = state["max_action"]
        self.gamma = state["gamma"]
        self.policy_update_freq = state["policy_update_freq"]


class CrossQTD3_Agent(Base_Agent):
    """
    Original Paper: https://arxiv.org/abs/1802.09477v3
    """

    def __init__(
        self,
        env: gym.Env,
        actor_hidden_layers: list[int] = [256, 256],
        critic_hidden_layers: list[int] = [256, 256],
        actor_lr: float = 3e-3,
        critic_lr: float = 3e-3,
        device: str = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        policy_freq: int = 2,
        replay_buffer=None,
        use_wandb=False,
    ):
        super().__init__(env, replay_buffer)

        self.env = env
        self.initial_training_steps = 1000
        self.training_steps_per_rollout = 1
        state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.device = device if device is not None else self.get_device()

        self.replay_buffer = None if replay_buffer is None else replay_buffer

        # initialize the actor and critic networks
        self.actor = Deterministic_Actor(
            state_dim, self.action_dim, env, actor_hidden_layers
        ).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)

        self.critic = CrossQCritic(
            state_dim, self.action_dim, critic_hidden_layers, activation="tanh"
        ).to(self.device)

        # Actor target netowrk is always used for evaluation only
        self.target_actor.eval()

        # define parameters
        self.max_action = env.action_space.high
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_freq = policy_freq

        # define optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(0.5, 0.999)
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(0.5, 0.999)
        )

        if use_wandb:
            wandb.init(
                project="crossq_td3_project",  # set this from config file
                config={
                    "actor_hidden_layers": actor_hidden_layers,
                    "critic_hidden_layers": critic_hidden_layers,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "max_action": self.max_action,
                    "gamma": gamma,
                    "tau": tau,
                    "policy_freq": policy_freq,
                },
            )

    def select_action(self, state: torch.Tensor, train: bool) -> torch.Tensor:
        """
        input: states (torch.Tensor)
        output: actions (torch.Tensor)
        """
        if train:
            self.actor.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action = self.actor(state).cpu().data.numpy().flatten()
                noise = torch.normal(
                    torch.Tensor([0]),
                    torch.Tensor(self.max_action * self.exploration_noise),
                    size=self.action_dim,
                )
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
                action = torch.Tensor(action)
            self.actor.train()

        else:
            self.target_actor.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action = self.target_actor(state)  #
                print(action)
                print(self.action_dim)
                noise = (
                    torch.normal(0.0, self.policy_noise, size=(self.action_dim,))
                    .clamp(-self.noise_clip, self.noise_clip)
                    .to(self.device)
                )
                action = (action + noise).cpu()
                action = (
                    action.clamp(
                        torch.Tensor([-self.max_action]),
                        torch.Tensor([self.max_action]),
                    )
                    .data.numpy()
                    .flatten()
                )
            self.target_actor.train()

        return action

    def train(
        self, rollout_eps: int, batch_size: int, total_steps: int, save_freq: int
    ) -> None:
        """
        Train the agent
        """

        for global_step in range(total_steps):
            if len(self.replay_buffer) == 0:
                self._do_random_actions(self.initial_training_steps)
            else:
                episode_reward, steps = self.rollout(rollout_eps, train=True)
                if self.use_wandb:
                    wandb.log(
                        {
                            "rollout/episode_reward": episode_reward,
                            "rollout/episode_lenght": steps,
                            "rollout/avg_reward": episode_reward / steps,
                        }
                    )

            state, action, reward, next_state, terminated, _= (
                self.replay_buffer.sample(batch_size)
            )

            # convert everything to tensors
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            action_tensor = torch.FloatTensor(action).to(self.device)
            reward_tensor = torch.FloatTensor(reward).to(self.device)
            terminated_tensor = torch.tensor(
                terminated, dtype=torch.float32, device=self.device
            )  # ? Are terminated and truncated boolean values?
            # truncated_tensor = torch.tensor(
            #     truncated, dtype=torch.float32, device=self.device
            # )  # ? do we need truncated values?

            # calculate the Q
            # noise = (torch.randn_like(action_tensor) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = torch.Tensor(self.select_action(next_state, train=False))

            state_next_state = torch.cat(
                [state_tensor, next_state_tensor], dim=0
            )  # * If state tensor is 1D, put an unsqueeze(0)
            action_next_action = torch.cat(
                [action_tensor, next_action], dim=0
            )  # * If state tensor is 1D, put an unsqueeze(0)

            q_vals_1, q_vals_2 = self.critic(state_next_state, action_next_action)
            q_curr_1, q_next_1 = torch.chunk(q_vals_1, chunks=2, dim=0)
            q_curr_2, q_next_2 = torch.chunk(q_vals_2, chunks=2, dim=0)

            target_q = torch.min(q_next_1, q_next_2)
            target_q = reward_tensor + (1 - terminated_tensor) * self.gamma * target_q

            # calculate and update critic loss
            torch.detach(target_q)

            critic_loss_1 = F.mse_loss(q_curr_1, target_q)
            critic_loss_2 = F.mse_loss(q_curr_2, target_q)

            total_critic_loss = critic_loss_1 + critic_loss_2

            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic_optimizer.step()

            # every N steps update the actor
            if global_step % self.policy_freq == 0:
                n_action = self.select_action(state_tensor, train=False)
                self.critic.eval()
                with torch.no_grad():
                    q_values_1, q_values_2 = self.critic(state_tensor, n_action)
                min_q_value = torch.min(q_values_1, q_values_2)
                actor_loss = -min_q_value.mean()
                self.critic.train()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update target networks
                for actor_param, target_param in zip(
                    self.actor.parameters(), self.target_actor.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * actor_param.data + (1 - self.tau) * target_param.data
                    )

            # log stuff
            if self.use_wandb and global_step % 10 == 0:
                # Compute gradient and do optimizer step logging #! might remove this
                critic_grad_norm = (
                    sum(
                        [
                            p.grad.data.norm(2).item() ** 2
                            for p in self.critic.parameters()
                        ]
                    )
                    ** 0.5
                )

                wandb.log(
                    {
                        "critic_1_loss": critic_loss_1,
                        "critic_2_loss": critic_loss_2,
                        "critic_loss": total_critic_loss,
                        "critic_grad_norm": critic_grad_norm,
                        "global_step": global_step,
                        "actor_loss": actor_loss,
                    }
                )
            # Save the model checkpoint every save_freq training steps
            if global_step % save_freq == 0 and global_step > 0:
                self.save(f"model_checkpoint_{global_step}.pt")

    def save(self, filename: str) -> None:
        models = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "gamma": self.gamma,
            "tau": self.tau,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "exploration_noise": self.exploration_noise,
            "policy_update_freq": self.policy_freq,
        }
        torch.save(models, filename)

    def load(self, filename: str) -> None:
        models = torch.load(filename)
        self.actor.load_state_dict(models["actor_state_dict"])
        self.critic.load_state_dict(models["critic_state_dict"])
        self.actor_optimizer.load_state_dict(models["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(models["critic_optimizer_state_dict"])
        self.gamma = models["gamma"]
        self.tau = models["tau"]
        self.policy_noise = models["policy_noise"]
        self.noise_clip = models["noise_clip"]
        self.exploration_noise = models["exploration_noise"]
        self.policy_freq = models["policy_update_freq"]
