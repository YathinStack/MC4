"""
Engine 2 — Stage 2: Kalman Tracker
=====================================
Maintains one 6-state Kalman Filter per tracked obstacle.
Smooths noisy position measurements and infers velocity.

State vector: [x, y, z, vx, vy, vz]   (6D)
Observation:  [x, y, z]                (3D — position only)

Classes
-------
ObstacleTracker     — single-obstacle KF with miss-frame counting
TrackerManager      — spawns / associates / removes ObstacleTrackers
"""
from __future__ import annotations

import uuid
import time
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

from engine2.sensor_simulator import Detection


# ── Hyper-parameters ──────────────────────────────────────────────────────────

DT              = 1.0 / 30.0     # 30 Hz update rate
MAX_MISS_FRAMES = 10             # frames without a detection → remove tracker
ASSOC_DIST_M    = 1.5            # max association distance (metres)
HISTORY_LEN     = 90             # store 3 s of trajectory history


# ── Single-obstacle Kalman Filter ─────────────────────────────────────────────

class ObstacleTracker:
    """
    6-state constant-velocity Kalman Filter for one physical obstacle.

    State:
        x = [x, y, z, vx, vy, vz]^T

    Transition model (constant velocity):
        F = | I₃  dt·I₃ |
            | 0₃     I₃ |

    Observation model (position only):
        H = [I₃  0₃]

    Noise:
        R = 0.05 · I₃   (sensor noise ~5 cm std dev)
        Q = 0.01 · I₆   (process noise — trust motion model strongly)
        P = 0.10 · I₆   (initial uncertainty)
    """

    def __init__(self,
                 initial_detection: Detection,
                 dt: float = DT):
        self.tracker_id    = str(uuid.uuid4())[:8]
        self.obs_id        = initial_detection.obstacle_id
        self.radius        = initial_detection.radius
        self.missed_frames = 0
        self.frame_count   = 0
        self.dt            = dt

        # Trajectory history for visualisation / debugging
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)

        # ── Build filterpy KalmanFilter ──────────────────────────────────────
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # Transition matrix — constant-velocity model
        self.kf.F = np.eye(6, dtype=np.float64)
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt

        # Observation matrix — position only
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance
        self.kf.R = np.eye(3, dtype=np.float64) * 0.05

        # Process noise covariance
        # Small noise for position, higher noise for velocity to learn quickly
        self.kf.Q = np.eye(6, dtype=np.float64)
        self.kf.Q[0:3, 0:3] *= 0.01
        self.kf.Q[3:6, 3:6] *= 1.0

        # Initial covariance
        self.kf.P = np.eye(6, dtype=np.float64)
        self.kf.P[0:3, 0:3] *= 0.1
        self.kf.P[3:6, 3:6] *= 10.0

        # Initial state — position from first detection, velocity zero
        p = initial_detection.position
        self.kf.x = np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0],
                              dtype=np.float64).reshape(6, 1)

        self.history.append(p.copy())

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self) -> None:
        """Run the KF prediction step (called once per tick even without a measurement)."""
        self.kf.predict()

    def update(self, detection: Detection) -> None:
        """
        Assimilate a new measurement into the filter.
        Also resets the miss-frame counter and updates radius estimate.
        """
        z = detection.position.reshape(3, 1)
        self.kf.update(z)
        self.radius        = 0.9 * self.radius + 0.1 * detection.radius  # EMA
        self.missed_frames = 0
        self.frame_count  += 1
        self.history.append(self.position.copy())

    def mark_missed(self) -> None:
        """Called when no detection was associated this tick."""
        self.missed_frames += 1

    @property
    def is_lost(self) -> bool:
        return self.missed_frames >= MAX_MISS_FRAMES

    @property
    def position(self) -> np.ndarray:
        """Current smoothed position estimate [x, y, z]."""
        return self.kf.x[:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate [vx, vy, vz]."""
        return self.kf.x[3:6, 0].copy()

    @property
    def speed(self) -> float:
        """Scalar speed (m/s)."""
        return float(np.linalg.norm(self.velocity))

    def __repr__(self) -> str:
        p, v = self.position, self.velocity
        return (f"<Tracker {self.tracker_id} obs={self.obs_id}"
                f" pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
                f" vel=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})"
                f" miss={self.missed_frames}>")


# ── Multi-obstacle tracker manager ────────────────────────────────────────────

class TrackerManager:
    """
    Manages a dynamic pool of ObstacleTrackers.

    Each tick:
    1. Predict all existing trackers.
    2. Associate incoming detections to nearest tracker (greedy NN matching).
    3. Update matched trackers; mark unmatched trackers as missed.
    4. Spawn new tracker for any unmatched detection.
    5. Remove trackers that exceeded MAX_MISS_FRAMES.
    """

    def __init__(self,
                 assoc_dist: float = ASSOC_DIST_M,
                 dt: float = DT):
        self.trackers: dict[str, ObstacleTracker] = {}
        self.assoc_dist = assoc_dist
        self.dt         = dt

    # ── Core tick ─────────────────────────────────────────────────────────────

    def update(self, detections: list[Detection]) -> None:
        """
        Process one batch of detections (one sensor frame).
        Modifies self.trackers in-place.
        """
        # Step 1 — predict all existing trackers forward
        for t in self.trackers.values():
            t.predict()

        # Step 2 — greedy nearest-neighbour association
        unmatched_detections = list(detections)
        matched_tracker_ids  = set()

        for det in detections:
            best_id   = None
            best_dist = self.assoc_dist

            for tid, tracker in self.trackers.items():
                if tid in matched_tracker_ids:
                    continue
                dist = float(np.linalg.norm(tracker.position - det.position))
                if dist < best_dist:
                    best_dist = dist
                    best_id   = tid

            if best_id is not None:
                self.trackers[best_id].update(det)
                matched_tracker_ids.add(best_id)
                unmatched_detections.remove(det)

        # Step 3 — mark unmatched trackers as missed
        for tid, tracker in self.trackers.items():
            if tid not in matched_tracker_ids:
                tracker.mark_missed()

        # Step 4 — spawn new trackers for unmatched detections
        for det in unmatched_detections:
            new_t = ObstacleTracker(det, dt=self.dt)
            self.trackers[new_t.tracker_id] = new_t

        # Step 5 — remove lost trackers
        lost = [tid for tid, t in self.trackers.items() if t.is_lost]
        for tid in lost:
            del self.trackers[tid]

    # ── Query ─────────────────────────────────────────────────────────────────

    def active_trackers(self) -> list[ObstacleTracker]:
        """Return all currently active (not-lost) trackers."""
        return list(self.trackers.values())

    def __len__(self) -> int:
        return len(self.trackers)

    def __repr__(self) -> str:
        return f"<TrackerManager n={len(self.trackers)} trackers>"
