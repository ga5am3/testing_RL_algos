import numpy as np

class Base_Agent:
    def __init__(self, env, replay_buffer, max_action):
        self.env = env
        self.replay_buffer = replay_buffer
        self.max_action = max_action
        
    def _do_random_actions(self, batch_size: int) -> None:
        actions = np.random.uniform(
            -self.max_action,
            self.max_action,
            size=(batch_size, self.env.action_space.shape[0])
        )

        states = self.env.reset(batch_size)
        for i in range(batch_size):
            state = self.env.reset()
            while not terminated or not truncated: #TODO: add behavior for truncated episodes
                action = actions[i] # While rollouts the target actor is used
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                state = next_state
    