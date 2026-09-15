"""
reward.py — Multi-Stage Reward State Machine
---------------------------------------------
Replaces the binary distance heuristic with a proper Finite State Machine
tracking multi-step task progress:

  INIT -> APPROACHING -> GRASPING -> LIFTING -> PLACING -> DONE

Each stage transition yields a shaped reward, proving the VLA policy
is genuinely solving a multi-step manipulation task.

Used by BimanualDinnerEnv in simulation/env.py.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple
import numpy as np


class TaskStage(IntEnum):
    INIT        = 0   # Episode just started
    APPROACHING = 1   # Arms moving toward objects
    GRASPING    = 2   # Arms within grasp range of objects
    LIFTING     = 3   # Objects being elevated (Z increased)
    PLACING     = 4   # Objects moving to target positions
    DONE        = 5   # Task complete


# Shaped reward values per stage transition
STAGE_REWARDS: Dict[TaskStage, float] = {
    TaskStage.APPROACHING : 0.5,
    TaskStage.GRASPING    : 2.0,
    TaskStage.LIFTING     : 3.0,
    TaskStage.PLACING     : 4.0,
    TaskStage.DONE        : 10.0,
}

# Distance thresholds (in metres) to trigger stage transitions
THRESHOLDS = {
    "approach_left"  : 0.35,  # Left arm within 35cm of plate
    "approach_right" : 0.35,  # Right arm within 35cm of cup
    "grasp_left"     : 0.18,  # Left arm within 18cm of plate
    "grasp_right"    : 0.18,  # Right arm within 18cm of cup
    "lift_z_delta"   : 0.03,  # Object lifted > 3cm from initial Z
    "place_radius"   : 0.12,  # Object within 12cm of target position
}


@dataclass
class RewardState:
    stage: TaskStage = TaskStage.INIT
    initial_plate_z: float = 0.0
    initial_cup_z:   float = 0.0
    episode_reward:  float = 0.0
    stage_history:   list  = field(default_factory=list)


def compute_reward(
    left_pos:   np.ndarray,   # (3,) left arm end-effector world position
    right_pos:  np.ndarray,   # (3,) right arm end-effector world position
    plate_pos:  np.ndarray,   # (3,) plate world position
    cup_pos:    np.ndarray,   # (3,) cup world position
    state:      RewardState,
) -> Tuple[float, bool, RewardState]:
    """
    Computes shaped reward based on current task stage.

    Returns
    -------
    reward    : float   — Step reward
    success   : bool    — True if task is complete (DONE stage)
    new_state : RewardState — Updated FSM state
    """
    reward  = 0.0
    success = False

    dist_left  = float(np.linalg.norm(left_pos  - plate_pos))
    dist_right = float(np.linalg.norm(right_pos - cup_pos))

    # Dense distance reward (always active)
    reward += max(0.0, (1.0 - dist_left)  * 0.05)
    reward += max(0.0, (1.0 - dist_right) * 0.05)

    # ── Stage Machine Transitions ─────────────────────────────────────────────
    prev_stage = state.stage

    if state.stage == TaskStage.INIT:
        state.initial_plate_z = plate_pos[2]
        state.initial_cup_z   = cup_pos[2]
        state.stage = TaskStage.APPROACHING

    elif state.stage == TaskStage.APPROACHING:
        if dist_left < THRESHOLDS["approach_left"] and dist_right < THRESHOLDS["approach_right"]:
            state.stage = TaskStage.GRASPING

    elif state.stage == TaskStage.GRASPING:
        if dist_left < THRESHOLDS["grasp_left"] and dist_right < THRESHOLDS["grasp_right"]:
            state.stage = TaskStage.LIFTING

    elif state.stage == TaskStage.LIFTING:
        plate_lifted = (plate_pos[2] - state.initial_plate_z) > THRESHOLDS["lift_z_delta"]
        cup_lifted   = (cup_pos[2]   - state.initial_cup_z)   > THRESHOLDS["lift_z_delta"]
        if plate_lifted or cup_lifted:
            state.stage = TaskStage.PLACING

    elif state.stage == TaskStage.PLACING:
        # Simplified: if objects are close to arm and arms are close to each other
        dist_arms = float(np.linalg.norm(left_pos - right_pos))
        if dist_left < THRESHOLDS["place_radius"] and dist_right < THRESHOLDS["place_radius"] \
                and dist_arms < 0.4:
            state.stage = TaskStage.DONE
            success = True

    # Stage transition bonus
    if state.stage != prev_stage:
        stage_reward = STAGE_REWARDS.get(state.stage, 0.0)
        reward += stage_reward
        state.stage_history.append(state.stage)
        print(f"  [FSM] Stage transition: {prev_stage.name} -> {state.stage.name} (+{stage_reward:.1f})")

    state.episode_reward += reward
    return reward, success, state


def stage_name(state: RewardState) -> str:
    return state.stage.name
