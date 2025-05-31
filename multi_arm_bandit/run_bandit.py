import argparse
from environment import MultiArmBandit
from explore import EpsGreedy, UCB
import numpy as np
import wandb
import matplotlib.pyplot as plt
import pandas as pd

steps = 50000

parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent",
    type=str,
    choices=["eps_greedy", "ucb", "both"],
    default="both",
    help="Agent type to use",
)
args = parser.parse_args()

wandb.init(project="bandit-regret", name="multi-agent-comparison")

seeds = [0, 1, 2, 3]
agents_to_run = ["eps_greedy", "ucb"] if args.agent == "both" else [args.agent]
all_results = {}

for agent_type in agents_to_run:
    all_regrets = []
    for seed in seeds:
        env = MultiArmBandit(k=10, seed=seed)
        if agent_type == "eps_greedy":
            agent = EpsGreedy(
                k=env.k,
                eps_start=0.1,
                eps_end=0.01,
                decay_steps=int(steps / 2),
                seed=seed,
            )
        elif agent_type == "ucb":
            agent = UCB(k=env.k, c=2.0, seed=seed)
        best_mean = env.q_star.max()
        cum_regret = 0
        regrets = []
        for i in range(steps):
            action = agent.act()
            reward = env.pull(action)
            agent.update(action, reward)
            cum_regret += best_mean - reward
            if i % 100 == 0:
                regrets.append(cum_regret)
        all_regrets.append(regrets)

    all_regrets = np.array(all_regrets)
    xs = list(range(0, steps, 100))
    mean = all_regrets.mean(axis=0)
    std = all_regrets.std(axis=0)

    all_results[agent_type] = {"xs": xs, "mean": mean, "std": std}

# Create a wandb Table for plotting
for agent_type, results in all_results.items():
    df = pd.DataFrame(
        {"step": results["xs"], "mean": results["mean"], "std": results["std"]}
    )
    table = wandb.Table(dataframe=df)
    wandb.log({f"{agent_type}_table": table})

# Create a combined plot
plt.figure(figsize=(8, 5))
for agent_type, color in zip(["eps_greedy", "ucb"], ["blue", "orange"]):
    mean = all_results[agent_type]["mean"]
    std = all_results[agent_type]["std"]
    xs = all_results[agent_type]["xs"]
    plt.plot(xs, mean, label=f"{agent_type} mean", color=color)
    plt.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

plt.xlabel("step")
plt.ylabel("cumulative regret")
plt.title("Cumulative Regret Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("regret_comparison.png")
wandb.log({"regret_comparison": wandb.Image("regret_comparison.png")})
plt.close()

wandb.finish()
