import numpy as np
import pytest
from secureslice.environment import NetworkSliceEnv

def test_capacity_constraint_and_reward_shape():
    env = NetworkSliceEnv(seed=3, capacity=2.0)
    env.reset()
    states, rewards, done, info = env.step([2, 2, 2])
    assert len(states) == 3
    assert rewards.shape == (3,)
    assert info["total_allocation"] <= 2.0 + 1e-9
    assert done is False

def test_rejects_invalid_actions():
    env = NetworkSliceEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step([0, 1])

def test_seeded_runs_are_reproducible():
    a, b = NetworkSliceEnv(seed=11), NetworkSliceEnv(seed=11)
    a.reset(); b.reset()
    out_a = a.step([1, 1, 1])
    out_b = b.step([1, 1, 1])
    assert np.allclose(out_a[1], out_b[1])
