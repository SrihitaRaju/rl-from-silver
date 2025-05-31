import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x

    def select_action(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        action = torch.multinomial(probs, 1).item()
        action_log_prob = log_probs[0, action]

        # Compute entropy for this state
        entropy = -(probs * log_probs).sum(dim=1)

        return action, action_log_prob, entropy
