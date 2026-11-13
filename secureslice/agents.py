from __future__ import annotations
import numpy as np

class IndependentQLearner:
    """Independent tabular Q-learning agent for one network slice."""
    def __init__(self, states: int = 18, actions: int = 3, alpha: float = 0.15,
                 gamma: float = 0.95, epsilon: float = 0.15, seed: int = 7):
        self.q = np.zeros((states, actions), dtype=float)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        return int(np.argmax(self.q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * float(np.max(self.q[next_state]))
        self.q[state, action] += self.alpha * (target - self.q[state, action])
