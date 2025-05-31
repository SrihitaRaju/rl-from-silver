import gymnasium as gym
from reinforce_agent import ReinforceAgent
import numpy as np


def train(env, agent, num_episodes=1000, gamma=0.99, entropy_coef=0.01):
    returns_history = []
    for episode in range(num_episodes):
        states, actions, rewards, log_probs, entropies = agent.collect_episode(env)
        returns = agent.compute_returns(rewards, gamma)
        agent.update_policy(returns, log_probs, entropies, entropy_coef)

        episode_return = sum(rewards)
        returns_history.append(episode_return)

        if episode % 10 == 0:
            avg_return = np.mean(returns_history[-10:])
            print(f"Episode {episode}, Average Return: {avg_return:.2f}")


def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    # Try these learning rates
    # learning_rates = [1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    learning_rates = [5e-3]
    use_baseline = True
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}, baseline: {use_baseline}")
        agent = ReinforceAgent(
            state_dim, action_dim, hidden_dim=128, lr=lr, use_baseline=use_baseline
        )
        train(env, agent)


if __name__ == "__main__":
    main()
