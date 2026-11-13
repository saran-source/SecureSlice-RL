# SecureSlice-RL

A research prototype for adaptive and security-aware 5G/6G network-slice resource allocation. This repository begins with a deterministic toy environment and tabular multi-agent Q-learning baseline so experiments can be run and tested without NS-3 or a GPU.

## Current baseline

- A configurable synthetic network-slicing environment.
- Multiple slice agents with local observations.
- Reward combining SLA satisfaction, resource efficiency and security violations.
- Tabular Q-learning baseline with reproducible random seeds.
- Episode runner and CSV metrics export.
- Unit tests for environment transitions and reproducibility.

This is a simulation baseline, not a validated 5G/6G deployment. Federated training, GNN topology encoding, PyTorch policies and NS-3 integration are roadmap items until implemented and benchmarked.

## Quick start

Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m secureslice.train --episodes 300 --seed 7
pytest -q
```

Results are written to `outputs/episode_metrics.csv`.

## Environment model

Each episode simulates three logical slices (eMBB, URLLC and mMTC) sharing a finite resource budget. Each agent chooses a discrete allocation adjustment. The environment samples traffic demand and security pressure from a seeded generator. The reward penalizes SLA shortfall, over-allocation and security exposure.

## Roadmap

- [ ] Add topology-aware graph observations and a GNN encoder.
- [ ] Add federated parameter aggregation and client heterogeneity experiments.
- [ ] Add PyTorch multi-agent policy optimization.
- [ ] Integrate an NS-3-backed network simulator adapter.
- [ ] Add adversarial traffic and attack scenarios with safe synthetic traces.
- [ ] Compare against static, greedy and centralized baselines.
- [ ] Report confidence intervals across multiple random seeds.

## Reproducibility

Record the seed, environment configuration, code revision and dependency versions for every experiment. Do not report patent status, SLA improvements or benchmark performance without supporting public evidence and reproducible measurements.
