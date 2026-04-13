# utils.py
import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax().item()