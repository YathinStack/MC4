"""
Phase 6 — RRT* Real-Time Replanner.

Finds minimum-cost collision-free paths in 3D using RRT* (Karaman & Frazzoli 2011).

Parameters: max 1000 samples, step 0.3 m, goal bias 0.2,
            rewire radius 1.0 m, 50 ms timeout.

Switched-Lyapunov safety condition:
    Safe to switch iff V(ξ_t) ≤ ρ_old  AND  V(ξ_t) ≤ (1−κ)^(−N) · ρ_new
    where N = 10 is the switch horizon.
"""
from __future__ import annotations

import time
import math
import random
from typing import Optional

import torch

from src import config as cfg


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _segment_sphere_collision(p1: list[float], p2: list[float],
                               center: list[float], radius: float) -> bool:
    """
    Test line-segment p1→p2 against sphere (center, radius).
    Uses quadratic discriminant method.
    """
    d = [p2[i] - p1[i] for i in range(3)]
    f = [p1[i] - center[i] for i in range(3)]

    a = sum(d[i]**2 for i in range(3))
    b = 2 * sum(f[i]*d[i] for i in range(3))
    c = sum(f[i]**2 for i in range(3)) - radius**2

    disc = b**2 - 4*a*c
    if disc < 0:
        return False

    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2*a + 1e-12)
    t2 = (-b + sq) / (2*a + 1e-12)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))


# ── RRT* Planner ──────────────────────────────────────────────────────────────

class RRTReplanner:
    """
    RRT* path planner for 3D obstacle environments.

    Usage:
        planner = RRTReplanner(workspace_bounds=[(-10,10)]*3)
        path, latency_ms = planner.plan(start, goal, obstacles)
    """

    def __init__(self,
                 workspace_bounds: list[tuple[float, float]] | None = None,
                 max_samples: int = cfg.RRT_MAX_SAMPLES,
                 step_size: float  = cfg.RRT_STEP_SIZE,
                 goal_bias: float  = cfg.RRT_GOAL_BIAS,
                 rewire_radius: float = cfg.RRT_REWIRE_RADIUS,
                 timeout_ms: float = cfg.RRT_TIMEOUT_MS):

        self.bounds       = workspace_bounds or [(-15.0, 15.0)] * 3
        self.max_samples  = max_samples
        self.step         = step_size
        self.goal_bias    = goal_bias
        self.rewire_r     = rewire_radius
        self.timeout_ms   = timeout_ms

    def _sample(self, goal: list[float]) -> list[float]:
        if random.random() < self.goal_bias:
            return goal[:]
        return [random.uniform(lo, hi) for lo, hi in self.bounds]

    def _steer(self, frm: list[float], to: list[float]) -> list[float]:
        d = _dist3(frm, to)
        if d < self.step:
            return to[:]
        scale = self.step / d
        return [frm[i] + scale*(to[i]-frm[i]) for i in range(3)]

    def _collision_free(self, p1: list[float], p2: list[float],
                        obstacles: list[dict]) -> bool:
        for obs in obstacles:
            r_safe = obs["radius"] + cfg.DRONE_RADIUS
            if _segment_sphere_collision(p1, p2, obs["center"], r_safe):
                return False
        return True

    def plan(self, start: list[float], goal: list[float],
             obstacles: list[dict]) -> tuple[list[list[float]], float]:
        """
        Run RRT* and return (path, latency_ms).

        path is a list of 3D waypoints from start to goal.
        If no path is found within timeout, returns [start, goal].
        """
        t0 = time.perf_counter()

        nodes  = [start[:]]
        parent = {0: -1}
        cost   = {0: 0.0}

        for _ in range(self.max_samples):
            if (time.perf_counter() - t0) * 1000 > self.timeout_ms:
                break

            q_rand = self._sample(goal)

            # Nearest neighbour
            nearest_idx = min(range(len(nodes)),
                              key=lambda i: _dist3(nodes[i], q_rand))
            q_new = self._steer(nodes[nearest_idx], q_rand)

            if not self._collision_free(nodes[nearest_idx], q_new, obstacles):
                continue

            # Near neighbours for rewiring
            near_idxs = [i for i, n in enumerate(nodes)
                         if _dist3(n, q_new) <= self.rewire_r]

            # Choose best parent
            best_parent = nearest_idx
            best_cost   = cost[nearest_idx] + _dist3(nodes[nearest_idx], q_new)
            for ni in near_idxs:
                c = cost[ni] + _dist3(nodes[ni], q_new)
                if c < best_cost and self._collision_free(nodes[ni], q_new, obstacles):
                    best_parent = ni
                    best_cost   = c

            new_idx          = len(nodes)
            nodes.append(q_new)
            parent[new_idx]  = best_parent
            cost[new_idx]    = best_cost

            # Rewire
            for ni in near_idxs:
                if ni == best_parent:
                    continue
                c = best_cost + _dist3(q_new, nodes[ni])
                if c < cost[ni] and self._collision_free(q_new, nodes[ni], obstacles):
                    parent[ni] = new_idx
                    cost[ni]   = c

            # Goal reached?
            if _dist3(q_new, goal) < self.step:
                # Reconstruct path
                path = [goal[:]]
                idx  = new_idx
                while idx != -1:
                    path.append(nodes[idx][:])
                    idx = parent[idx]
                path.reverse()
                latency = (time.perf_counter() - t0) * 1000
                return path, latency

        latency = (time.perf_counter() - t0) * 1000
        return [start[:], goal[:]], latency


# ── Switched Lyapunov Safety Check ────────────────────────────────────────────

def switched_lyapunov_safe(V_curr: float,
                            rho_old: float,
                            rho_new: float,
                            kappa: float = cfg.KAPPA,
                            N: int = 10) -> bool:
    """
    Check Switched Lyapunov condition for trajectory switching.

    Safe to switch iff:
        V(ξ_t) ≤ ρ_old   AND   V(ξ_t) ≤ (1−κ)^{-N} · ρ_new
    """
    return (V_curr <= rho_old) and (V_curr <= (1.0 - kappa)**(-N) * rho_new)
