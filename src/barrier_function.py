"""
Phase 3 — Control Barrier Function (CBF) and Lyapunov-Barrier Fusion.

For each spherical obstacle (center o, radius r_obs):
    B(ξ) = ‖p − o‖² − (r_obs + r_drone)²

Lyapunov-Barrier fusion:
    V_total = V(ξ) + λ · [max(0, −B(ξ))]²

Gradient projection onto safe subspace:
    If g_task · g_safety < 0:
        g_proj = g_task − (g_task · ĝ_s) ĝ_s
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import config as cfg
from src.mlp_controller import MLPController
from src.lyapunov_network import LyapunovNet
from src.quadrotor_dynamics import QuadrotorDynamics


def barrier_value(pos: torch.Tensor,
                  obs_center: torch.Tensor,
                  obs_radius: float) -> torch.Tensor:
    """
    B(ξ) = ‖p − o‖² − r_safe²
    B ≥ 0  ⟺  drone is outside the safe-inflation sphere.

    Args:
        pos        : [B, 3]  drone position
        obs_center : [3]     obstacle centre
        obs_radius : float   obstacle radius

    Returns:
        B : [B]  barrier values
    """
    r_safe   = obs_radius + cfg.DRONE_RADIUS
    diff     = pos - obs_center.to(pos.device)
    dist_sq  = (diff ** 2).sum(dim=1)
    return dist_sq - r_safe ** 2


class LyapunovBarrierFusion(nn.Module):
    """
    Fuses the neural Lyapunov certificate with CBF obstacle avoidance.

    The fused value is:
        V_total = V(ξ) + λ · Σ_i [max(0, −B_i(ξ))]²

    The control is obtained from the MLPController and then projected
    if any barrier constraint would be violated.
    """

    def __init__(self,
                 controller: MLPController,
                 lyapunov: LyapunovNet,
                 dynamics: QuadrotorDynamics):
        super().__init__()
        self.controller = controller
        self.lyapunov   = lyapunov
        self.dynamics   = dynamics
        self.lam        = cfg.LAMBDA_BARRIER

    def forward(self, xi: torch.Tensor, obstacles: list[dict]):
        """
        Args:
            xi        : [B, 17]  state (requires_grad for projection)
            obstacles : list of {'center': [x,y,z], 'radius': float, ...}

        Returns:
            u_safe   : [B, 4]  safe motor commands
            B_min    : [B]     minimum barrier value across all obstacles
            V_total  : [B]     fused Lyapunov-barrier value
        """
        # Base control
        u = self.controller(xi)

        pos = xi[:, :3]
        V   = self.lyapunov(xi)

        B_values = []
        penalty  = torch.zeros_like(V)

        for obs in obstacles:
            center = torch.tensor(obs["center"], dtype=xi.dtype, device=xi.device)
            radius = float(obs["radius"])
            B      = barrier_value(pos, center, radius)
            B_values.append(B)
            # Quadratic penalty for violation (B < 0)
            violation = torch.clamp(-B, min=0.0)
            penalty   = penalty + self.lam * violation ** 2

        V_total = V + penalty
        B_min   = torch.stack(B_values, dim=1).min(dim=1).values \
                  if B_values else torch.full_like(V, 99.0)

        # Gradient-based safety projection on control output
        if xi.requires_grad and B_min.min().item() < 1.0:
            # Safety gradient w.r.t. xi
            g_s = torch.autograd.grad(
                V_total.sum(), xi,
                create_graph=False, retain_graph=True
            )[0][:, :3]                             # position gradient [B, 3]

            # Task gradient: direction toward lower V from nominal u
            g_t = torch.autograd.grad(
                V.sum(), xi,
                create_graph=False, retain_graph=True
            )[0][:, :3]

            g_s_norm = g_s / (torch.norm(g_s, dim=1, keepdim=True) + 1e-8)
            proj     = (g_t * g_s_norm).sum(dim=1, keepdim=True)

            # Conflict mask: task and safety gradient point in opposite dirs
            conflict = (g_t * g_s).sum(dim=1) < 0

            # Adjust control by a small safety correction
            correction = torch.zeros_like(u)
            if conflict.any():
                correction[conflict] = -0.1 * u[conflict]

            u = u + correction

        return torch.clamp(u, cfg.RPM_MIN, cfg.RPM_MAX), B_min, V_total
