import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from reinforce_agent import ReinforceAgent


def train_agent(env, agent, num_episodes):
    """Train agent and return episode rewards"""
    episode_rewards = []
    reward_buffer = deque(maxlen=100)

    for episode in range(num_episodes):
        # Collect episode
        states, actions, rewards, log_probs, mean_entropy = agent.collect_episode(env)
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        reward_buffer.append(episode_reward)

        # Compute returns and update
        returns = agent.compute_returns(rewards)
        agent.update_policy(returns, log_probs)

        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(reward_buffer)
            print(f"Episode {episode}: Average reward = {avg_reward:.2f}")
        if avg_reward > 475:
            print(f"Solved in {episode} episodes")
            # pad rewards with 0s to make it 1000 episodes
            episode_rewards = episode_rewards + [avg_reward] * (
                1000 - len(episode_rewards)
            )
            break

    return episode_rewards


def plot_comparison(rewards_baseline, rewards_no_baseline, window=50):
    """Plot rewards with moving average and variance bands"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Moving average function
    def moving_average(data, window):
        ret = np.cumsum(data, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        return ret[window - 1 :] / window

    # Calculate rolling statistics
    def rolling_stats(data, window):
        means = []
        stds = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            end = i + 1
            window_data = data[start:end]
            means.append(np.mean(window_data))
            stds.append(np.std(window_data))
        return np.array(means), np.array(stds)

    episodes = np.arange(len(rewards_baseline))

    # Calculate statistics
    mean_baseline, std_baseline = rolling_stats(rewards_baseline, window)
    mean_no_baseline, std_no_baseline = rolling_stats(rewards_no_baseline, window)

    # Plot mean lines
    ax.plot(episodes, mean_baseline, label="With Baseline", color="blue", linewidth=2)
    ax.plot(
        episodes, mean_no_baseline, label="Without Baseline", color="red", linewidth=2
    )

    # Plot variance bands (±1 std)
    ax.fill_between(
        episodes,
        mean_baseline - std_baseline,
        mean_baseline + std_baseline,
        alpha=0.3,
        color="blue",
    )
    ax.fill_between(
        episodes,
        mean_no_baseline - std_no_baseline,
        mean_no_baseline + std_no_baseline,
        alpha=0.3,
        color="red",
    )

    # Formatting
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(
        "REINFORCE: With vs Without Baseline\n(Moving Average ± Std Dev)", fontsize=14
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add solved threshold
    ax.axhline(y=195, color="green", linestyle="--", alpha=0.7, label="Solved")

    plt.tight_layout()
    plt.savefig("reinforce_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Train with baseline
    print("Training WITH baseline...")
    agent_baseline = ReinforceAgent(
        state_dim, action_dim, hidden_dim=128, use_baseline=True, lr=2e-3
    )
    rewards_baseline = train_agent(env, agent_baseline, num_episodes=1000)

    # Reset environment seed for fair comparison
    env = gym.make("CartPole-v1")

    # Train without baseline
    print("\nTraining WITHOUT baseline...")
    agent_no_baseline = ReinforceAgent(
        state_dim, action_dim, hidden_dim=128, use_baseline=False, lr=2e-3
    )
    rewards_no_baseline = train_agent(env, agent_no_baseline, num_episodes=1000)

    # Plot results
    plot_comparison(rewards_baseline, rewards_no_baseline)

    # Print final statistics
    print(f"\nFinal 100-episode statistics:")
    print(
        f"With baseline - Mean: {np.mean(rewards_baseline[-100:]):.2f}, Std: {np.std(rewards_baseline[-100:]):.2f}"
    )
    print(
        f"Without baseline - Mean: {np.mean(rewards_no_baseline[-100:]):.2f}, Std: {np.std(rewards_no_baseline[-100:]):.2f}"
    )


if __name__ == "__main__":
    main()
