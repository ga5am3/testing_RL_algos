import torch
import numpy as np

class Base_Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        
    def _do_random_actions(self, initial_steps: int) -> None:
        # Sample random actions depending on the type of action space
        actions = np.array([self.env.action_space.sample() for _ in range(initial_steps)])

        for i in range(initial_steps):
            state, _ = self.env.reset()
            terminated, truncated = False, False  
            while not terminated and not truncated:  
                action = actions[i]  # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, terminated, truncated)
                state = next_state

    def rollout(self, eval: bool) -> None:
        """
        Rollout the agent in the environment
        """
        # print("Rollout step ", _)
        state, _ = self.env.reset(seed=0)  # TODO: make seed a parameter
        termination = False
        truncation = False

        total_ep_reward = 0
        steps = 0
        while (not termination) and (not truncation):
            action = self.select_action(state, eval)
            next_state, reward, termination, truncation, infos = self.env.step(action)
            # print(reward)
            # if self.use_wandb:
            #     wandb.log({
            #         "rollout/reward_per_step": reward
            #     })
            self.replay_buffer.add(
                state, action, reward, next_state, termination, truncation
            )
            state = next_state
            steps += 1
            total_ep_reward += reward

        # print(f"Episode finished in {steps} steps with Average Reward = {total_ep_reward:.2f}")
        # if self.use_wandb:
        # wandb.log({
        #     "rollout/episode_steps": steps,
        #     "rollout/avg_reward": total_ep_reward/steps,
        #     "rollout/episode_reward": total_ep_reward
        # })
        return total_ep_reward, steps

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    