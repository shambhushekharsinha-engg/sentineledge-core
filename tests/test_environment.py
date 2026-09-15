"""Tests for BimanualDinnerEnv simulation environment."""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_env_import():
    from simulation.env import BimanualDinnerEnv
    assert BimanualDinnerEnv is not None


def test_env_reset_returns_correct_keys():
    from simulation.env import BimanualDinnerEnv
    env = BimanualDinnerEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    assert "pixels" in obs
    assert "state" in obs
    assert "instruction" in obs
    assert isinstance(info, dict)
    env.close()


def test_env_observation_shapes():
    from simulation.env import BimanualDinnerEnv
    env = BimanualDinnerEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert obs["pixels"].shape == (480, 640, 3), f"Unexpected pixel shape: {obs['pixels'].shape}"
    assert obs["state"].shape == (16,), f"Unexpected state shape: {obs['state'].shape}"
    assert isinstance(obs["instruction"], str)
    assert len(obs["instruction"]) > 0
    env.close()


def test_env_step_returns_correct_types():
    from simulation.env import BimanualDinnerEnv
    env = BimanualDinnerEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "is_success" in info
    env.close()


def test_env_domain_randomization_varies_across_seeds():
    from simulation.env import BimanualDinnerEnv
    env = BimanualDinnerEnv(render_mode="rgb_array")
    obs0, _ = env.reset(seed=0)
    obs1, _ = env.reset(seed=99)
    # Different seeds should produce different observations due to randomization
    assert not np.array_equal(obs0["pixels"], obs1["pixels"]), \
        "Domain randomization should produce different observations for different seeds"
    env.close()


def test_env_custom_instruction():
    from simulation.env import BimanualDinnerEnv
    env = BimanualDinnerEnv(render_mode="rgb_array")
    custom = "Place the cup on the plate."
    obs, _ = env.reset(seed=0, options={"instruction": custom})
    assert obs["instruction"] == custom
    env.close()


def test_env_action_space_shape():
    from simulation.env import BimanualDinnerEnv
    env = BimanualDinnerEnv(render_mode="rgb_array")
    assert env.action_space.shape == (16,)
    action = env.action_space.sample()
    assert action.shape == (16,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    env.close()
