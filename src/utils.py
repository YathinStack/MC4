"""
Shared utility functions used across training, testing, and the dashboard.
"""
from __future__ import annotations

import math
import torch

from src import config as cfg


# ── Unit conversions ──────────────────────────────────────────────────────────

def rpm_to_rads(rpm: torch.Tensor | float) -> torch.Tensor | float:
    """Convert RPM to rad/s."""
    return rpm * (2.0 * math.pi / 60.0)


def rads_to_rpm(rads: torch.Tensor | float) -> torch.Tensor | float:
    """Convert rad/s to RPM."""
    return rads * (60.0 / (2.0 * math.pi))


# ── State packing / unpacking ─────────────────────────────────────────────────

def unpack_state(xi: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Unpack a batched state tensor [B, 17] into named components.

    Returns
    -------
    dict with keys: pos, vel, quat, omega, motors
    """
    return {
        "pos":    xi[:, 0:3],
        "vel":    xi[:, 3:6],
        "quat":   xi[:, 6:10],
        "omega":  xi[:, 10:13],
        "motors": xi[:, 13:17],
    }


# ── Geometry ──────────────────────────────────────────────────────────────────

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion [B, 4] = (w, x, y, z) to rotation matrix [B, 3, 3].
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2),
    ], dim=1).reshape(-1, 3, 3)

    return R


# ── Energy computations ───────────────────────────────────────────────────────

def compute_kinetic_energy(vel: torch.Tensor,
                            omega: torch.Tensor,
                            mass: float,
                            J: torch.Tensor) -> torch.Tensor:
    """
    KE = ½ m ||v||² + ½ ωᵀ J ω

    Args:
        vel   : [B, 3]
        omega : [B, 3]
        mass  : float
        J     : [3, 3]  inertia tensor

    Returns:
        KE : [B]
    """
    KE_trans = 0.5 * mass * (vel ** 2).sum(dim=1)
    Jw = omega @ J.to(omega.device).t()
    KE_rot   = 0.5 * (omega * Jw).sum(dim=1)
    return KE_trans + KE_rot


def compute_potential_energy(pos: torch.Tensor, mass: float) -> torch.Tensor:
    """
    PE = m g z   (ENU frame — z is altitude)

    Args:
        pos  : [B, 3]
        mass : float

    Returns:
        PE : [B]
    """
    return mass * cfg.GRAVITY_SEA_LEVEL * pos[:, 2]


# ── Random state generators ────────────────────────────────────────────────────

def random_near_equilibrium(batch: int = 1, radius: float = 1.0) -> torch.Tensor:
    """
    Sample random states uniformly inside an L2 ball of given radius
    around the hover equilibrium.

    Returns : [batch, 17]
    """
    xi = cfg.EQUILIBRIUM_STATE.unsqueeze(0).repeat(batch, 1).clone()
    noise = torch.randn(batch, cfg.STATE_REPR_DIM)
    noise = noise / noise.norm(dim=1, keepdim=True)          # unit shell
    scale = torch.rand(batch, 1) ** (1.0 / cfg.STATE_REPR_DIM) * radius
    xi = xi + noise * scale
    # Re-normalise quaternion
    xi[:, 6:10] /= xi[:, 6:10].norm(dim=1, keepdim=True)
    # Clamp motors
    xi[:, 13:17] = torch.clamp(xi[:, 13:17], cfg.RPM_MIN, cfg.RPM_MAX)
    return xi


# ── Physics validation ────────────────────────────────────────────────────────

def validate_hover_rpm() -> dict:
    """
    Validate that HOVER_RPM produces exactly mg thrust.

    Returns dict with hover_rpm, thrust_total, weight, error_pct.
    """
    hover_rad   = rpm_to_rads(cfg.HOVER_RPM)
    thrust_each = cfg.K_F * hover_rad ** 2
    thrust_total = 4.0 * thrust_each
    weight       = cfg.DRONE_MASS * cfg.GRAVITY_SEA_LEVEL
    error_pct    = abs(thrust_total - weight) / weight * 100.0

    return {
        "hover_rpm":    cfg.HOVER_RPM,
        "thrust_total": thrust_total,
        "weight":       weight,
        "error_pct":    error_pct,
    }
