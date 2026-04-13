# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import ContinuousMazeEnv
from DQN_model import DQN
from utils import ReplayBuffer, select_action

# Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 5000
TARGET_UPDATE = 20
MAX_EPISODES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dqn():
    env = ContinuousMazeEnv(render_mode=None)
    policy_net = DQN(input_dim=2, output_dim=4).to(DEVICE)
    target_net = DQN(input_dim=2, output_dim=4).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    epsilon = EPSILON_START

    consecutive_successes = 0

    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        # Enable rendering every 100 episodes
        if episode % 100 == 0:
            env.render_mode = "human"
        else:
            env.render_mode = None

        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space, DEVICE)
            next_state, reward, done, _, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + (1 - dones) * GAMMA * next_q_values

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if env.render_mode == "human":
                env.render()

        if info["distance_to_goal"] <= env.goal_radius:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Steps: {step_count}, "
              f"Epsilon: {epsilon:.3f}, Successes: {consecutive_successes}")

        if consecutive_successes >= 100 and epsilon <= EPSILON_MIN:
            print("Training complete: 100 consecutive successes at epsilon=0.1")
            break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), "dqn_model.pth")
    env.close()
    return policy_net

if __name__ == "__main__":
    train_dqn()