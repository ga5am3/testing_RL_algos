import numpy as np
import gymnasium as gym
from utils.buffers import SimpleBuffer
import torch
from algos.CrossQ import CrossQSAC_Agent


def do_random_actions(env, replay_buffer, max_action, batch_size: int) -> None:
            
    actions = np.array([env.action_space.sample() for _ in range(batch_size)])

    for i in range(batch_size):
        state = env.reset()
        terminated, truncated = False, False  # initialize termination flags
        while not terminated and not truncated:
            action = actions[i]
            next_state, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(state, next_state, action, reward, terminated, truncated)
            state = next_state

def test_rollout(agent, buffer):
    env = gym.make('Pendulum-v1', disable_env_checker=True)
    #buffer = SimpleBuffer(max_size=1500, batch_size=32, gamma=0.99, n_steps=1, seed=0)
    
    # Import the agent from your project.
    from algos.CrossQ import CrossQSAC_Agent
    agent = CrossQSAC_Agent(env, replay_buffer=buffer)
    
    # Override select_action with a dummy policy that returns a random action.
    # Note: The dummy action is wrapped in a tensor and unsqueezed to mimic the original output shape.
    agent.select_action = lambda state, train: torch.tensor(env.action_space.sample()).unsqueeze(0)
    
    # Run a short rollout (e.g., max_steps=2) for testing purposes.
    agent.rollout(env, episodes=4, eval=True)
    print("Stored transitions after rollout:", len(buffer))

def test_experience(agent):
    # Sample some transitions from the agent's replay buffer.
    try:
        states, actions, rewards, next_states, terminated, truncated = agent.replay_buffer.sample(2)
        print("Sampled experience:")
        print("States:", np.array(states))
        print("Actions:", np.array(actions))
        print("Rewards:", np.array(rewards))
        print("Next States:", np.array(next_states))
        print("Terminated Flags:", terminated)
        print("Truncated Flags:", truncated)
    except Exception as e:
        print("Error while sampling experience:", e)    

if __name__ == "__main__":
    # Dummy replay buffer for testing purposes
    env = gym.make('Pendulum-v1')
    max_action = 1.0  # adjust based on your env's action space
    batch_size = 2


    buffer = SimpleBuffer(max_size=1500, batch_size=32, gamma=0.99, n_steps=10, seed=0)
    #do_random_actions(env, buffer, max_action, batch_size)
    print("Stored transitions:", len(buffer))

    agent = CrossQSAC_Agent(env, replay_buffer=buffer)
    test_rollout(agent, buffer)
    print(len(agent.replay_buffer))
    print(len(buffer))
    test_experience(agent)