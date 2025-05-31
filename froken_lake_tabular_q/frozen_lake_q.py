import gymnasium as gym
import numpy as np
from typing import Tuple
from decay_schedule import decay_schedule
import argparse


class FrozenLakeRL:
    def __init__(
        self,
        env_name: str = "FrozenLake-v1",
        is_slippery: bool = True,
        max_steps: int = 1000,
    ):
        self.env = gym.make(
            env_name, is_slippery=is_slippery, max_episode_steps=max_steps
        )
        # self.Q = np.zeros(self.env.observation_space.n, self.env.action_space.n)
        # TODO: think about max steps per episode
        # hyperparameters
        self.gamma = 0.9
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

    def visualize_grid(self):
        """Visualize the Frozen Lake grid showing holes, frozen surface, start and goal positions."""
        # Get the grid layout from the environment
        grid = self.env.desc

        # Convert bytes to strings and create a readable representation
        grid_str = []
        for row in grid:
            row_str = []
            for cell in row:
                if cell == b"S":
                    row_str.append("S")  # Start
                elif cell == b"F":
                    row_str.append("F")  # Frozen
                elif cell == b"H":
                    row_str.append("H")  # Hole
                elif cell == b"G":
                    row_str.append("G")  # Goal
            grid_str.append(row_str)

        # Print the grid with a border
        print("\nFrozen Lake Grid Layout:")
        print("Legend: S=Start, F=Frozen, H=Hole, G=Goal")
        print("+" + "-" * (len(grid_str[0]) * 2) + "+")
        for row in grid_str:
            print("|" + " ".join(row) + "|")
        print("+" + "-" * (len(grid_str[0]) * 2) + "+")

    def value_iteration(
        self, theta: float = 1e-8, max_iterations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements value iteration algorithm to compute optimal value function V* and action-value function Q*.

        Args:
            gamma: Discount factor
            theta: Convergence threshold
            max_iterations: Maximum number of iterations

        Returns:
            V: Optimal state value function
            Q: Optimal action-value function
        """
        V = np.zeros(self.n_states)
        Q = np.zeros((self.n_states, self.n_actions))
        for iteration in range(max_iterations):
            V_old = V.copy()
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    transitions = self.env.unwrapped.P[s][a]
                    q_sa = 0
                    for prob, next_state, reward, done in transitions:
                        q_sa += prob * (
                            reward + (self.gamma * V[next_state] * (1 - int(done)))
                        )
                    Q[s, a] = q_sa
            V = np.max(Q, axis=1)
            if np.max(np.abs(V - V_old)) < theta:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        return V, Q

    def extract_policy(self, Q: np.ndarray) -> np.ndarray:
        return np.argmax(Q, axis=1)

    def q_learning(
        self,
        n_episodes: int = 50000,
        alpha_start: float = 0.5,
        alpha_end: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_ratio: float = 0.8,
        convergence_check: bool = True,
        convergence_window: int = 100,
        convergence_threshold: float = 0.01,
        max_steps: int = 1000,
    ) -> Tuple[np.ndarray, list]:
        """
        Implements Q-learning algorithm to learn the optimal Q-function.

        Args:
            n_episodes: Number of episodes to run
            alpha_start: Initial learning rate
            alpha_end: Final learning rate
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            decay_ratio: Proportion of episodes over which to decay epsilon
            convergence_check: Whether to check for early convergence
            convergence_window: Number of episodes to check for convergence
            convergence_threshold: Max Q-value change threshold for convergence
            max_steps: Maximum number of steps per episode

        Returns:
            Q: Learned Q-function
            rewards: List of rewards for each episode
        """
        Q = np.zeros((self.n_states, self.n_actions))
        episode_rewards = []
        Q_old = Q.copy()

        epsilon_schedule = decay_schedule(
            init_value=epsilon_start,
            min_value=epsilon_end,
            decay_ratio=decay_ratio,
            max_steps=n_episodes,
        )
        alpha_schedule = decay_schedule(
            init_value=alpha_start,
            min_value=alpha_end,
            decay_ratio=decay_ratio,
            max_steps=n_episodes,
        )

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            current_epsilon = epsilon_schedule[episode]
            current_alpha = alpha_schedule[episode]
            done = False
            steps = 0

            while not done and steps < max_steps:
                # epsilon greedy action selection
                if np.random.random() < current_epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(Q[state])

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                best_next_q = np.max(Q[next_state]) if not terminated else 0
                Q[state, action] = Q[state, action] + current_alpha * (
                    reward + self.gamma * best_next_q - Q[state, action]
                )

                state = next_state
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)

            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:])
                print(
                    f"Episode {episode + 1}, Avg Reward (last 1000): {avg_reward:.3f}, Epsilon: {current_epsilon:.3f}"
                )

            # Check for convergence
            if convergence_check and episode >= convergence_window:
                # Check Q-value stability
                q_change = np.max(np.abs(Q - Q_old))

                # Check performance
                recent_avg = np.mean(episode_rewards[-convergence_window:])

                # Check if we've converged
                if (
                    q_change < convergence_threshold
                    and recent_avg > 0.7
                    and current_epsilon <= epsilon_end
                ):
                    print(f"\nConverged after {episode + 1} episodes!")
                    print(
                        f"Q-value change: {q_change:.6f}, Avg reward: {recent_avg:.3f}"
                    )
                    break

                # Update Q_old for next convergence check
                if episode % convergence_window == 0:
                    Q_old = Q.copy()

        return Q, episode_rewards

    def evaluate_policy(
        self, policy: np.ndarray, n_episodes: int = 100, max_steps: int = 1000
    ) -> float:
        """
        Evaluate a policy by running multiple episodes.

        Args:
            policy: Policy to evaluate (array mapping states to actions)
            n_episodes: Number of evaluation episodes
            max_steps: Maximum number of steps per episode

        Returns:
            Average reward per episode
        """
        total_rewards = 0
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            while not done and steps < max_steps:
                action = policy[state]
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            total_rewards += episode_reward
        return total_rewards / n_episodes

    def render_policy(self, policy: np.ndarray):
        """Render the policy as a grid of actions."""
        action_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}

        if self.n_states == 16:  # 4x4 grid
            grid_size = 4
        elif self.n_states == 64:  # 8x8 grid
            grid_size = 8
        else:
            print(f"Policy rendering not supported for {self.n_states} states")
            return

        print("\nPolicy visualization:")
        for row in range(grid_size):
            for col in range(grid_size):
                state = row * grid_size + col
                print(action_symbols[policy[state]], end=" ")
            print()


def main(
    env_name: str = "FrozenLake-v1", is_slippery: bool = True, max_steps: int = 1000
):
    # Create environment
    rl = FrozenLakeRL(env_name=env_name, is_slippery=is_slippery, max_steps=max_steps)

    # Visualize the grid first
    rl.visualize_grid()

    print("\n=== Value Iteration ===")
    V_optimal, Q_optimal = rl.value_iteration()
    policy_vi = rl.extract_policy(Q_optimal)

    print("\nOptimal Value Function:")
    grid_size = int(np.sqrt(rl.n_states))
    print(V_optimal.reshape(grid_size, grid_size))

    print("\nOptimal Policy:")
    rl.render_policy(policy_vi)

    # Evaluate value iteration policy
    vi_reward = rl.evaluate_policy(policy_vi)
    print(f"\nValue Iteration Policy Average Reward: {vi_reward:.3f}")

    print("\n=== Q-Learning ===")
    Q_learned, rewards = rl.q_learning(n_episodes=50000)
    policy_ql = rl.extract_policy(Q_learned)

    print("\nLearned Policy:")
    rl.render_policy(policy_ql)

    # Evaluate Q-learning policy
    ql_reward = rl.evaluate_policy(policy_ql)
    print(f"\nQ-Learning Policy Average Reward: {ql_reward:.3f}")

    # Compare Q-functions
    print("\nQ-function difference (L2 norm):", np.linalg.norm(Q_optimal - Q_learned))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Frozen Lake RL algorithms")
    parser.add_argument(
        "--no-slippery",
        action="store_true",
        help="Run in non-slippery mode (default: slippery)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake-v1",
        choices=["FrozenLake-v1", "FrozenLake8x8-v1"],
        help="Environment to use (default: FrozenLake-v1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode (default: 1000)",
    )
    args = parser.parse_args()

    main(env_name=args.env, is_slippery=not args.no_slippery, max_steps=args.max_steps)
