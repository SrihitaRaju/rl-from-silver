import gymnasium as gym
import wandb
import os

# Set offline mode for CI environments
if "GITHUB_ACTIONS" in os.environ:
    os.environ["WANDB_MODE"] = "offline"

wandb.init(project="rl_bootstrap", name="random_cartpole")
env = gym.make("CartPole-v1")

for ep in range(10):
    obs, _ = env.reset()
    done, total = False, 0
    while not done:
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total += r
        done = terminated or truncated
    wandb.log({"reward": total}, step=ep)
    print(f"Episode {ep}: Reward = {total}")

env.close()
wandb.finish()
