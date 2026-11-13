from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SliceConfig:
    name: str
    min_rate: float
    weight: float
    security_weight: float

DEFAULT_SLICES = (
    SliceConfig("embb", 0.55, 1.0, 0.2),
    SliceConfig("urllc", 0.75, 1.8, 1.0),
    SliceConfig("mmtc", 0.30, 0.7, 0.4),
)

class NetworkSliceEnv:
    """Small synthetic shared-capacity environment for algorithm prototyping."""
    def __init__(self, seed: int = 7, capacity: float = 2.0, slices=DEFAULT_SLICES):
        self.rng = np.random.default_rng(seed)
        self.capacity = float(capacity)
        self.slices = tuple(slices)
        self.n_agents = len(self.slices)
        self.action_values = np.array([-0.1, 0.0, 0.1], dtype=float)
        self.allocations = np.full(self.n_agents, self.capacity / self.n_agents)
        self.demand = np.array([0.7, 0.9, 0.4], dtype=float)
        self.security_pressure = np.zeros(self.n_agents)
        self.steps = 0

    def reset(self):
        self.allocations[:] = self.capacity / self.n_agents
        self.demand = self.rng.uniform(0.25, 1.0, self.n_agents)
        self.security_pressure = self.rng.uniform(0.0, 0.4, self.n_agents)
        self.steps = 0
        return self.observations()

    def observations(self):
        return [
            np.array([self.allocations[i] / self.capacity, self.demand[i], self.security_pressure[i]], dtype=float)
            for i in range(self.n_agents)
        ]

    def _state(self, i: int) -> int:
        alloc = min(2, int(self.allocations[i] / self.capacity * 3))
        demand = min(2, int(self.demand[i] * 3))
        risk = min(1, int(self.security_pressure[i] * 2))
        return alloc * 6 + demand * 2 + risk

    def step(self, actions):
        if len(actions) != self.n_agents or any(int(a) not in (0, 1, 2) for a in actions):
            raise ValueError("one action in {0,1,2} is required per agent")
        before = [self._state(i) for i in range(self.n_agents)]
        delta = np.array([self.action_values[int(a)] for a in actions])
        proposed = np.maximum(0.05, self.allocations + delta)
        total = proposed.sum()
        self.allocations = proposed * min(1.0, self.capacity / total)
        self.demand = np.clip(self.demand + self.rng.normal(0, 0.08, self.n_agents), 0.1, 1.2)
        self.security_pressure = np.clip(self.security_pressure + self.rng.normal(0, 0.05, self.n_agents), 0, 1)
        rewards = []
        for i, cfg in enumerate(self.slices):
            served = min(self.allocations[i], self.demand[i])
            sla_shortfall = max(0.0, cfg.min_rate * self.demand[i] - served)
            security_penalty = cfg.security_weight * self.security_pressure[i] * self.allocations[i]
            efficiency_penalty = max(0.0, self.allocations[i] - self.demand[i]) * 0.2
            rewards.append(cfg.weight * served - 2.0 * sla_shortfall - security_penalty - efficiency_penalty)
        self.steps += 1
        done = self.steps >= 100
        after = [self._state(i) for i in range(self.n_agents)]
        info = {"allocations": self.allocations.copy(), "demand": self.demand.copy(),
                "security_pressure": self.security_pressure.copy(), "total_allocation": float(self.allocations.sum())}
        return after, np.asarray(rewards), done, info
