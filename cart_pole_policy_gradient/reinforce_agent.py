import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from policy_nn import PolicyNN


class ReinforceAgent:
    # What all should this class do?
    # initialize the policy network
    # run full episodes
    # store rewards and compute returns
    # keep running mean of returns
    def __init__(self, state_dim, action_dim, hidden_dim, use_baseline=True, lr=1e-3):
        self.policy_net = PolicyNN(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.baseline = 0
        self.use_baseline = use_baseline

    def collect_episode(self, env):
        # Run one episode, store (state, action, reward, log_prob, entropy)
        states, actions, rewards, log_probs, entropies = [], [], [], [], []
        state, info = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, entropy = self.policy_net.select_action(state_tensor)
            next_state, reward, done, truncated, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            state = next_state
        return states, actions, rewards, log_probs, entropies

    def compute_returns(self, rewards, gamma=0.99):
        # Monte Carlo returns: G_t = sum(gamma^k * r_{t+k})
        G_t = 0
        returns = []
        for r in reversed(rewards):
            G_t = r + gamma * G_t
            returns.append(G_t)
        returns = returns[::-1]

        # Convert to tensor and normalize (reduces variance)
        returns = torch.FloatTensor(returns)
        return returns
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8) #to center the returns around 0 not really necessary
        # return returns

    def update_policy(self, returns, log_probs, entropies, entropy_coef=0.01):
        # REINFORCE gradient step with entropy regularization
        policy_loss = []
        for log_prob, return_val, entropy in zip(log_probs, returns, entropies):
            policy_loss.append(-log_prob * return_val - entropy_coef * entropy)
        loss = torch.stack(policy_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
