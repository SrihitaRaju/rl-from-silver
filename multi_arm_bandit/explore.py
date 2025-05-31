import numpy as np


class EpsGreedy:
    def __init__(
        self, k: int = 10, eps_start=1.0, eps_end=0.1, decay_steps=6000, seed: int = 0
    ):
        self.k = k
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps
        self.rng = np.random.default_rng(seed)
        self.q_values = np.zeros(k)
        self.n_pulls = np.zeros(k)
        self.t = 0

    def act(self):
        self.t += 1
        best = np.argmax(self.q_values)
        eps_current = self.eps(self.t)
        if self.rng.random() < eps_current:
            return self.rng.integers(self.k)
        return best

    def eps(self, t: int):
        frac = min(1.0, t / self.decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def update(self, arm: int, reward: float):
        self.n_pulls[arm] += 1
        self.q_values[arm] += (1 / self.n_pulls[arm]) * (reward - self.q_values[arm])


class UCB:
    def __init__(self, k: int = 10, c: float = 2.0, seed: int = 0):
        self.k = k
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.q_values = np.zeros(k)
        self.n_pulls = np.zeros(k)
        self.t = 0

    def act(self):
        self.t += 1
        unvisited = np.where(self.n_pulls == 0)[0]
        if len(unvisited) > 0:
            return unvisited[0]  # Pull an unvisited arm
        u_values = self.c * np.sqrt(2 * np.log(self.t) / self.n_pulls)
        return np.argmax(self.q_values + u_values)

    def update(self, arm: int, reward: float):
        self.n_pulls[arm] += 1
        self.q_values[arm] += (1 / self.n_pulls[arm]) * (reward - self.q_values[arm])
