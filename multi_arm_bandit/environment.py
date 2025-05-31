import numpy as np


class MultiArmBandit:
    def __init__(self, k: int = 10, seed: int = 0):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.standard_normal(k)  # true means

    def pull(self, arm: int):
        return self.rng.normal(self.q_star[arm], 1)
