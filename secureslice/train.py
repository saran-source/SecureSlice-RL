from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from .environment import NetworkSliceEnv
from .agents import IndependentQLearner

def train(episodes: int = 300, seed: int = 7, output: str = "outputs/episode_metrics.csv"):
    env = NetworkSliceEnv(seed=seed)
    agents = [IndependentQLearner(seed=seed + i) for i in range(env.n_agents)]
    rows = []
    for episode in range(episodes):
        states = env.reset()
        totals = np.zeros(env.n_agents)
        done = False
        while not done:
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            next_states, rewards, done, info = env.step(actions)
            for i, agent in enumerate(agents):
                agent.update(states[i], actions[i], float(rewards[i]), next_states[i], done)
            totals += rewards
            states = next_states
        rows.append({"episode": episode + 1, "mean_reward": float(totals.mean()),
                     "total_reward": float(totals.sum()), "total_allocation": info["total_allocation"]})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["episode", "mean_reward", "total_reward", "total_allocation"])
        writer.writeheader()
        writer.writerows(rows)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/episode_metrics.csv")
    args = parser.parse_args()
    rows = train(args.episodes, args.seed, args.output)
    print(f"Completed {len(rows)} episodes. Metrics: {args.output}")

if __name__ == "__main__":
    main()
