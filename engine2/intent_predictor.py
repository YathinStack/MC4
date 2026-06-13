"""
Engine 2 — Stage 3: Intent Predictor
======================================
Predicts where each tracked obstacle will be T seconds in the future
using linear extrapolation of the Kalman-estimated velocity.

Phase 1: linear constant-velocity prediction.
Future phases could add learned intent models (e.g., social force models,
LSTM trajectory predictors) behind the same interface.

    predicted_center = position + velocity * T
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from engine2.kalman_tracker import ObstacleTracker


# ── Prediction result ─────────────────────────────────────────────────────────

@dataclass
class ObstaclePrediction:
    """All information about one tracked obstacle, ready for the shared bus."""
    tracker_id:        str
    obs_id:            str
    current_position:  np.ndarray    # [x, y, z] — KF-smoothed
    predicted_center:  np.ndarray    # [x, y, z] — T seconds ahead
    velocity:          np.ndarray    # [vx, vy, vz] m/s
    radius:            float         # metres
    horizon_s:         float         # prediction horizon used


# ── Predictor ─────────────────────────────────────────────────────────────────

class IntentPredictor:
    """
    Converts Kalman-tracker state into forward-predicted obstacle positions.

    Parameters
    ----------
    horizon_s : float
        How many seconds ahead to predict (default 0.5 s — gives Engine 1
        a 500 ms head-start to react before the obstacle arrives).
    clamp_speed : float
        If the KF velocity estimate exceeds this (m/s), cap it before
        predicting.  Prevents runaway extrapolation from a noisy burst.
        Default: 10.0 m/s (faster than any relevant obstacle).
    """

    def __init__(self,
                 horizon_s:   float = 0.5,
                 clamp_speed: float = 10.0):
        self.horizon_s   = horizon_s
        self.clamp_speed = clamp_speed

    # ── Core prediction ───────────────────────────────────────────────────────

    def predict_one(self, tracker: ObstacleTracker) -> ObstaclePrediction:
        """
        Predict the future position of a single tracked obstacle.

        Formula (linear extrapolation):
            v_clamped        = v * min(1, clamp_speed / |v|)
            predicted_center = p + v_clamped * T
        """
        pos = tracker.position          # [3]  current KF estimate
        vel = tracker.velocity          # [3]  current KF velocity

        # Clamp speed to prevent wild extrapolations
        speed = float(np.linalg.norm(vel))
        if speed > self.clamp_speed:
            vel = vel * (self.clamp_speed / speed)

        predicted = pos + vel * self.horizon_s

        return ObstaclePrediction(
            tracker_id       = tracker.tracker_id,
            obs_id           = tracker.obs_id,
            current_position = pos,
            predicted_center = predicted,
            velocity         = vel.copy(),
            radius           = tracker.radius,
            horizon_s        = self.horizon_s,
        )

    def predict_all(self,
                    trackers: list[ObstacleTracker]) -> list[ObstaclePrediction]:
        """Run predict_one for all active trackers."""
        return [self.predict_one(t) for t in trackers]

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def prediction_error(prediction: ObstaclePrediction,
                         actual_future_position: np.ndarray) -> float:
        """
        Euclidean error between predicted center and where the obstacle
        actually ends up T seconds later.  Used in test scripts only.
        """
        return float(np.linalg.norm(
            prediction.predicted_center - actual_future_position
        ))
