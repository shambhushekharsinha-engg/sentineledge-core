"""
action_filter.py — Real-time Action Smoothing for Physical Robots
-----------------------------------------------------------------
Raw neural network outputs can be noisy or jerky, which damages physical 
motors (like Dynamixels or SO-101 joints). This module implements a 
Low-Pass Exponential Moving Average (EMA) filter to ensure smooth, 
safe kinematic execution on the hardware.

Usage:
    filter = ActionFilter(action_dim=16, alpha=0.2)
    smooth_action = filter.process(raw_action)
"""

import numpy as np

class ActionFilter:
    def __init__(self, action_dim: int, alpha: float = 0.2):
        """
        Parameters
        ----------
        action_dim : int
            Dimension of the action space (16 for dual SO-101 arms).
        alpha : float
            Smoothing factor between 0.0 and 1.0.
            Lower = smoother but more delay.
            Higher = more responsive but less smoothing.
        """
        self.action_dim = action_dim
        self.alpha = alpha
        self.prev_action = None

    def reset(self):
        """Resets the filter state (e.g., at the start of a new episode)."""
        self.prev_action = None

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Applies EMA smoothing to the raw action.
        
        Parameters
        ----------
        raw_action : np.ndarray
            Shape (action_dim,) from the VLA policy.
            
        Returns
        -------
        np.ndarray
            Smoothed action of the same shape.
        """
        raw_action = np.array(raw_action, dtype=np.float32)
        
        if self.prev_action is None:
            self.prev_action = raw_action
            return raw_action

        # EMA Formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        smoothed = self.alpha * raw_action + (1.0 - self.alpha) * self.prev_action
        self.prev_action = smoothed
        
        return smoothed
