"""Tests for the reward state machine."""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_reward_import():
    from simulation.reward import compute_reward, RewardState, TaskStage
    assert compute_reward is not None


def test_reward_initial_stage():
    from simulation.reward import compute_reward, RewardState, TaskStage
    state = RewardState()
    assert state.stage == TaskStage.INIT


def test_reward_approaching_transition():
    from simulation.reward import compute_reward, RewardState, TaskStage
    state = RewardState()
    # Arms far away — should move to APPROACHING but not GRASPING
    left_pos  = np.array([0.0, 0.0, 0.9])
    right_pos = np.array([0.5, 0.0, 0.9])
    plate_pos = np.array([0.0, 0.1, 0.8])
    cup_pos   = np.array([0.3, 0.2, 0.8])
    reward, success, state = compute_reward(left_pos, right_pos, plate_pos, cup_pos, state)
    assert state.stage.value >= 1   # At least APPROACHING
    assert isinstance(reward, float)
    assert success is False


def test_reward_success_on_close_arms():
    from simulation.reward import compute_reward, RewardState, TaskStage
    state = RewardState()
    # Simulate arms very close to objects — should progress quickly
    plate_pos = np.array([0.0, 0.0, 0.8])
    cup_pos   = np.array([0.1, 0.0, 0.8])

    for _ in range(20):
        left_pos  = plate_pos + np.array([0.01, 0.0, 0.0])
        right_pos = cup_pos   + np.array([0.01, 0.0, 0.0])
        reward, success, state = compute_reward(
            left_pos, right_pos, plate_pos, cup_pos, state
        )

    assert state.stage.value > TaskStage.APPROACHING


def test_reward_dense_reward_positive():
    from simulation.reward import compute_reward, RewardState
    state = RewardState()
    plate_pos = np.array([0.0, 0.0, 0.8])
    cup_pos   = np.array([0.1, 0.0, 0.8])
    left_pos  = plate_pos + np.array([0.05, 0.0, 0.0])
    right_pos = cup_pos   + np.array([0.05, 0.0, 0.0])
    reward, _, _ = compute_reward(left_pos, right_pos, plate_pos, cup_pos, state)
    assert reward >= 0.0, "Dense reward should be non-negative when close"


def test_episode_reward_accumulates():
    from simulation.reward import compute_reward, RewardState
    state = RewardState()
    plate_pos = np.array([0.0, 0.0, 0.8])
    cup_pos   = np.array([0.1, 0.0, 0.8])
    for _ in range(5):
        left  = plate_pos + np.random.uniform(-0.2, 0.2, 3)
        right = cup_pos   + np.random.uniform(-0.2, 0.2, 3)
        compute_reward(left, right, plate_pos, cup_pos, state)
    assert state.episode_reward >= 0.0
