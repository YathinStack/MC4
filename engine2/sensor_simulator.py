"""
Engine 2 — Stage 1: Sensor Simulator
=====================================
Simulates moving obstacle detections at 30 Hz without real hardware.
Produces noisy 3-D position readings, exactly as a depth camera or LiDAR
would.  Replace this file with a RealSense SDK reader in Phase 3 — the
rest of Engine 2 stays identical.

Three obstacle archetypes
--------------------------
A  Linear mover   — constant velocity straight-line trajectory
B  Circular mover — person walking a horizontal circle
C  Random walk    — unpredictable obstacle (bird / balloon)
"""
from __future__ import annotations

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Detection dataclass ────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single raw obstacle detection from the sensor."""
    obstacle_id: str           # 'A', 'B', 'C'
    position: np.ndarray       # [x, y, z]  noisy measurement, metres (ENU)
    radius: float              # estimated obstacle radius, metres
    timestamp: float           # wall-clock seconds


# ── Base obstacle ──────────────────────────────────────────────────────────────

class _BaseObstacle:
    """Abstract simulated obstacle.  Subclasses implement `true_position(t)`."""

    def __init__(self, obs_id: str, radius: float, noise_sigma: float,
                 rng: np.random.Generator):
        self.obs_id      = obs_id
        self.radius      = radius
        self.noise_sigma = noise_sigma
        self.rng         = rng

    def true_position(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def true_velocity(self, t: float) -> np.ndarray:
        """Numerical velocity estimate (used only by test scripts for error measurement)."""
        dt  = 1e-4
        p1  = self.true_position(t + dt)
        p0  = self.true_position(t - dt)
        return (p1 - p0) / (2 * dt)

    def measure(self, t: float) -> Detection:
        """Return a noisy position measurement."""
        noise = self.rng.normal(0.0, self.noise_sigma, size=3)
        pos   = self.true_position(t) + noise
        return Detection(
            obstacle_id = self.obs_id,
            position    = pos.astype(np.float64),
            radius      = self.radius,
            timestamp   = t,
        )


# ── Type A — Linear mover ─────────────────────────────────────────────────────

class LinearObstacle(_BaseObstacle):
    """
    Moves at a constant velocity along a straight line.
    After reaching max_dist from origin it reverses direction (bounce).

    Default: starts at [3, 2, 1], velocity = [0.5, 0.0, 0.0] m/s
    Noise sigma = 0.05 m
    """

    def __init__(self,
                 obs_id:    str   = "A",
                 origin:    List[float] = None,
                 velocity:  List[float] = None,
                 radius:    float = 0.30,
                 noise_sigma: float = 0.05,
                 rng: np.random.Generator = None):
        super().__init__(obs_id, radius, noise_sigma,
                         rng or np.random.default_rng(0))
        self.origin   = np.array(origin   or [3.0, 2.0, 1.0], dtype=np.float64)
        self.velocity = np.array(velocity or [0.5, 0.0, 0.0],  dtype=np.float64)

    def true_position(self, t: float) -> np.ndarray:
        return self.origin + self.velocity * t


# ── Type B — Circular mover ───────────────────────────────────────────────────

class CircularObstacle(_BaseObstacle):
    """
    Moves in a horizontal circle — models a person walking a loop.

    Default: centre [0, 5, 1], radius_circle=2 m, angular velocity=0.3 rad/s
    Noise sigma = 0.08 m
    """

    def __init__(self,
                 obs_id:         str   = "B",
                 circle_center:  List[float] = None,
                 circle_radius:  float = 2.0,
                 angular_vel:    float = 0.3,    # rad/s
                 height:         float = 1.0,
                 radius:         float = 0.35,
                 noise_sigma:    float = 0.08,
                 rng: np.random.Generator = None):
        super().__init__(obs_id, radius, noise_sigma,
                         rng or np.random.default_rng(1))
        self.cx    = np.array(circle_center or [0.0, 5.0, 0.0], dtype=np.float64)
        self.r     = circle_radius
        self.omega = angular_vel
        self.z     = height

    def true_position(self, t: float) -> np.ndarray:
        theta = self.omega * t
        x     = self.cx[0] + self.r * math.cos(theta)
        y     = self.cx[1] + self.r * math.sin(theta)
        z     = self.cx[2] + self.z
        return np.array([x, y, z], dtype=np.float64)


# ── Type C — Random walk ──────────────────────────────────────────────────────

class RandomWalkObstacle(_BaseObstacle):
    """
    Each time step adds a random velocity perturbation — models unpredictable
    obstacles such as birds, balloons, or erratic pedestrians.

    Uses pre-generated trajectory so true_position(t) is reproducible and
    ground-truth error can be computed.

    Noise sigma = 0.10 m on measurements; velocity perturbation sigma = 0.2 m/s.
    """

    def __init__(self,
                 obs_id:       str   = "C",
                 origin:       List[float] = None,
                 vel_sigma:    float = 0.20,     # m/s per axis per second
                 dt:           float = 1.0 / 30, # simulation timestep
                 max_t:        float = 120.0,    # pre-generate this many seconds
                 radius:       float = 0.25,
                 noise_sigma:  float = 0.10,
                 rng: np.random.Generator = None):
        super().__init__(obs_id, radius, noise_sigma,
                         rng or np.random.default_rng(2))
        self.dt       = dt
        origin_arr    = np.array(origin or [-2.0, 3.0, 1.5], dtype=np.float64)

        # Pre-generate ground-truth trajectory
        n_steps = int(max_t / dt) + 2
        traj    = np.zeros((n_steps, 3), dtype=np.float64)
        vel     = np.zeros(3, dtype=np.float64)
        traj[0] = origin_arr
        for i in range(1, n_steps):
            vel     = vel * 0.9 + self.rng.normal(0, vel_sigma * dt, size=3)
            traj[i] = traj[i - 1] + vel * dt

        self._traj    = traj
        self._max_idx = n_steps - 1

    def true_position(self, t: float) -> np.ndarray:
        idx = min(int(t / self.dt), self._max_idx)
        return self._traj[idx].copy()


# ── Sensor Simulator ──────────────────────────────────────────────────────────

class SensorSimulator:
    """
    Aggregates all three obstacle types and emits a list of Detections
    on every call to `sense()`.

    Usage
    -----
        sim = SensorSimulator()
        sim.start()
        while True:
            detections = sim.sense()   # list[Detection], one per obstacle
            time.sleep(1/30)
    """

    RATE_HZ: float = 30.0

    def __init__(self, seed: int = 42):
        rng  = np.random.default_rng(seed)
        rng_a = np.random.default_rng(seed + 1)
        rng_b = np.random.default_rng(seed + 2)
        rng_c = np.random.default_rng(seed + 3)

        self._obstacles: List[_BaseObstacle] = [
            LinearObstacle(rng=rng_a),
            CircularObstacle(rng=rng_b),
            RandomWalkObstacle(rng=rng_c),
        ]
        self._t0: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Record the simulation start time."""
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        """Seconds since start()."""
        return time.perf_counter() - self._t0

    def sense(self) -> List[Detection]:
        """
        Return one noisy Detection per simulated obstacle,
        stamped with the current elapsed time.
        """
        t = self.elapsed()
        return [obs.measure(t) for obs in self._obstacles]

    def true_positions(self, t: float | None = None) -> dict[str, np.ndarray]:
        """
        Return ground-truth positions (no noise) — used only for error measurement
        in test scripts.  Never called by Engine 2 production code.
        """
        if t is None:
            t = self.elapsed()
        return {obs.obs_id: obs.true_position(t) for obs in self._obstacles}

    def true_velocities(self, t: float | None = None) -> dict[str, np.ndarray]:
        """Return ground-truth velocities — for test/validation only."""
        if t is None:
            t = self.elapsed()
        return {obs.obs_id: obs.true_velocity(t) for obs in self._obstacles}


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = SensorSimulator()
    sim.start()
    print("Sensor simulator demo — 5 samples at 30 Hz")
    for i in range(5):
        dets = sim.sense()
        for d in dets:
            print(f"  [{d.obstacle_id}] pos={d.position.round(3)}  r={d.radius}")
        time.sleep(1.0 / SensorSimulator.RATE_HZ)
