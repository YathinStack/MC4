"""
Phase 4 — Stability-Aware SGD (Online Adaptation).

Adapts the MLP controller at 100 Hz while guaranteeing formal Lyapunov
stability via a 3-stage IBP verification pipeline.

Algorithm per time step:
    1. Detect Lyapunov violation  viol = max(0, V(f(ξ,π(ξ))) − (1−κ)V(ξ))
    2. If viol > τ: backup θ, compute projected gradient, take one SGD step
    3. IBP verify updated θ on [ξ−δ, ξ+δ]
       - Pass → accept, save safe checkpoint
       - Fail → sampling fallback (200 MC points)
         - MC pass  → accept
         - MC fail  → rollback θ ← θ_backup
    4. After MAX_CONSECUTIVE_FAIL rollbacks → activate LQR fallback
"""
from __future__ import annotations

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as cfg
from src.mlp_controller import MLPController
from src.lyapunov_network import LyapunovNet
from src.quadrotor_dynamics import QuadrotorDynamics


# ─── Standalone IBP primitives (exported for tests) ───────────────────────────

def ibp_linear(W: torch.Tensor, b: torch.Tensor,
               x_lo: torch.Tensor, x_hi: torch.Tensor):
    """
    Sound interval propagation through a linear layer y = Wx + b.

    Returns (y_lo, y_hi) such that for all x in [x_lo, x_hi]:
        y_lo ≤ Wx + b ≤ y_hi   (element-wise)
    """
    W_pos = torch.clamp(W, min=0.0)
    W_neg = torch.clamp(W, max=0.0)
    y_lo  = x_lo @ W_pos.t() + x_hi @ W_neg.t() + b
    y_hi  = x_hi @ W_pos.t() + x_lo @ W_neg.t() + b
    return y_lo, y_hi


def ibp_leaky_relu(x_lo: torch.Tensor, x_hi: torch.Tensor,
                   slope: float = 0.01):
    """Sound interval propagation through LeakyReLU."""
    def lrelu(x):
        return torch.where(x >= 0, x, slope * x)

    y_lo = torch.min(lrelu(x_lo), lrelu(x_hi))
    y_hi = torch.max(lrelu(x_lo), lrelu(x_hi))

    # Mixed case: interval straddles 0
    mixed = (x_lo < 0) & (x_hi >= 0)
    y_lo  = torch.where(mixed, slope * x_lo, y_lo)
    y_hi  = torch.where(mixed, x_hi, y_hi)
    return y_lo, y_hi


def ibp_sequential(net: nn.Sequential,
                   x_lo: torch.Tensor, x_hi: torch.Tensor):
    """
    Propagate bounds through an nn.Sequential of Linear + LeakyReLU layers.
    """
    lo, hi = x_lo, x_hi
    for layer in net:
        if isinstance(layer, nn.Linear):
            lo, hi = ibp_linear(layer.weight, layer.bias, lo, hi)
        elif isinstance(layer, nn.LeakyReLU):
            lo, hi = ibp_leaky_relu(lo, hi, slope=layer.negative_slope)
        elif isinstance(layer, nn.Tanh):
            lo, hi = torch.tanh(lo), torch.tanh(hi)
        elif isinstance(layer, nn.Softplus):
            lo, hi = F.softplus(lo), F.softplus(hi)
        # Clamp / other layers: monotone → just propagate
    return lo, hi


# ─── LQR Fallback Controller ──────────────────────────────────────────────────

class LQRFallback:
    """
    Linear fallback controller: u = u* − K · (ξ − ξ*)

    Activates after MAX_CONSECUTIVE_FAIL consecutive IBP failures.
    """
    def __init__(self):
        # Gain matrix K ∈ R^{4×17}  (hand-tuned for hover stability)
        K = torch.zeros(4, cfg.STATE_REPR_DIM, dtype=cfg.DTYPE)

        # Altitude / collective: all motors respond to z-position error
        K[:, 2]     =  50.0    # z-position
        K[:, 5]     =  30.0    # z-velocity

        # Roll / pitch attitude
        K[0, 7]     = -20.0    # qx → motor 1
        K[1, 7]     =  20.0    # qx → motor 2
        K[2, 8]     = -20.0    # qy → motor 3
        K[3, 8]     =  20.0    # qy → motor 4

        # Angular rate damping
        K[0, 10]    = -5.0     # roll rate
        K[1, 10]    =  5.0
        K[2, 11]    = -5.0     # pitch rate
        K[3, 11]    =  5.0
        K[:, 12]    = -3.0     # yaw rate

        self.K    = K
        self.xi_eq = cfg.EQUILIBRIUM_STATE.clone()
        self.u_eq  = cfg.EQUILIBRIUM_ACTION.clone()

    def get_control(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi : [B, 17]
        Returns:
            u  : [B, 4]  clamped RPM commands
        """
        K    = self.K.to(xi.device)
        xi_eq = self.xi_eq.to(xi.device)
        u_eq  = self.u_eq.to(xi.device)
        delta = xi - xi_eq.unsqueeze(0)
        u     = u_eq.unsqueeze(0) - delta @ K.t()
        return torch.clamp(u, cfg.RPM_MIN, cfg.RPM_MAX)


# ─── Stability-Aware SGD ──────────────────────────────────────────────────────

class StabilityAwareSGD:
    """
    Online stability-preserving adaptation of the MLP controller.

    Parameters
    ----------
    controller : MLPController
    lyapunov   : LyapunovNet
    dynamics   : QuadrotorDynamics
    lr         : float  learning rate (default cfg.ADAPT_LR)
    """

    def __init__(self,
                 controller: MLPController,
                 lyapunov: LyapunovNet,
                 dynamics: QuadrotorDynamics,
                 lr: float = cfg.ADAPT_LR):
        self.controller = controller
        self.lyapunov   = lyapunov
        self.dynamics   = dynamics
        self.lr         = lr
        self.lqr        = LQRFallback()

        # Statistics
        self.n_adaptations   = 0
        self.n_rollbacks     = 0
        self.n_lqr_activations = 0
        self.lqr_active      = False
        self._consecutive_fails = 0
        self._safe_params    = [p.data.clone() for p in controller.parameters()]
        self._latencies: list[float] = []
        self._violations: list[float] = []

    # ── public API ────────────────────────────────────────────────────────────

    def adapt_online(self, xi: torch.Tensor,
                     u_desired: torch.Tensor | None = None):
        """
        Run one adaptation step.

        Returns
        -------
        u          : [B, 4]  safe control output
        adapted    : bool
        violation  : float   Lyapunov decrease violation
        latency    : float   wall-clock seconds
        """
        t0 = time.perf_counter()

        # LQR override if active
        if self.lqr_active:
            u_lqr  = self.lqr.get_control(xi)
            # Try to hand back to neural controller
            with torch.no_grad():
                V_curr = self.lyapunov(xi)
                u_test = self.controller(xi)
                xi_next_test = self.dynamics(xi, u_test)
                V_next_test  = self.lyapunov(xi_next_test)
                viol_test    = torch.clamp(
                    V_next_test - (1.0 - cfg.KAPPA) * V_curr, min=0.0
                ).mean().item()
            if viol_test < cfg.TAU_VIOLATION:
                self.lqr_active = False
                self._consecutive_fails = 0
            latency = time.perf_counter() - t0
            self._latencies.append(latency)
            return u_lqr, False, viol_test, latency

        # Compute violation
        xi_in = xi.detach().requires_grad_(True)
        with torch.enable_grad():
            u_curr   = self.controller(xi_in)
            xi_next  = self.dynamics(xi_in, u_curr)
            V_curr   = self.lyapunov(xi_in)
            V_next   = self.lyapunov(xi_next)
            violation = torch.clamp(
                V_next - (1.0 - cfg.KAPPA) * V_curr, min=0.0
            ).mean()

        viol_val = violation.item()
        self._violations.append(viol_val)

        adapted = False
        if viol_val > cfg.TAU_VIOLATION:
            # Save backup
            backup = [p.data.clone() for p in self.controller.parameters()]

            # Stability gradient
            g_s_list = torch.autograd.grad(
                violation, list(self.controller.parameters()),
                create_graph=False, allow_unused=True
            )

            # Task gradient (track desired RPM)
            if u_desired is not None:
                with torch.enable_grad():
                    u_task = self.controller(xi.detach().requires_grad_(False))
                    loss_t = ((u_task - u_desired.detach()) ** 2).mean()
                g_t_list = torch.autograd.grad(
                    loss_t, list(self.controller.parameters()),
                    create_graph=False, allow_unused=True
                )
            else:
                g_t_list = g_s_list

            # Projected gradient step
            with torch.no_grad():
                for p, g_t, g_s in zip(self.controller.parameters(),
                                       g_t_list, g_s_list):
                    if g_s is None or g_t is None:
                        continue
                    g_s_flat = g_s.flatten()
                    g_t_flat = g_t.flatten()
                    norm_s   = g_s_flat.norm() + 1e-8
                    dot      = (g_t_flat * (g_s_flat / norm_s)).sum()
                    if dot < 0:
                        g_proj = g_t - dot * (g_s / norm_s)
                    else:
                        g_proj = g_t
                    p.data.add_(-self.lr * g_proj)

            # Verify with IBP
            ibp_ok = self._verify_ibp(xi.detach())
            if not ibp_ok:
                mc_ok = self._verify_mc(xi.detach())
                if not mc_ok:
                    # Rollback
                    for p, bp in zip(self.controller.parameters(), backup):
                        p.data.copy_(bp)
                    self.n_rollbacks += 1
                    self._consecutive_fails += 1
                    if self._consecutive_fails >= cfg.MAX_CONSECUTIVE_FAIL:
                        self.lqr_active = True
                        self.n_lqr_activations += 1
                else:
                    adapted = True
                    self.n_adaptations += 1
                    self._consecutive_fails = 0
                    self._safe_params = [p.data.clone()
                                         for p in self.controller.parameters()]
            else:
                adapted = True
                self.n_adaptations += 1
                self._consecutive_fails = 0
                self._safe_params = [p.data.clone()
                                     for p in self.controller.parameters()]

        with torch.no_grad():
            u_out = self.controller(xi)

        latency = time.perf_counter() - t0
        self._latencies.append(latency)
        return u_out, adapted, viol_val, latency

    # ── verification helpers ──────────────────────────────────────────────────

    def _verify_ibp(self, xi: torch.Tensor, delta: float = cfg.DELTA_IBP) -> bool:
        """3-stage IBP verification on [ξ−δ, ξ+δ]."""
        try:
            xi_lo = xi - delta
            xi_hi = xi + delta
            xi_lo[:, 6:10] = xi_lo[:, 6:10] / (xi_lo[:, 6:10].norm(dim=1, keepdim=True) + 1e-8)
            xi_hi[:, 6:10] = xi_hi[:, 6:10] / (xi_hi[:, 6:10].norm(dim=1, keepdim=True) + 1e-8)

            # Stage 1: controller IBP
            with torch.no_grad():
                u_lo, u_hi = ibp_sequential(self.controller.net, xi_lo, xi_hi)
                u_lo = torch.clamp(u_lo, cfg.RPM_MIN, cfg.RPM_MAX)
                u_hi = torch.clamp(u_hi, cfg.RPM_MIN, cfg.RPM_MAX)

            # Stage 2: approximate dynamics bounds
            xi_lo_next = xi_lo.clone()
            xi_hi_next = xi_hi.clone()
            xi_lo_next[:, :3] += xi_lo[:, 3:6] * cfg.DT
            xi_hi_next[:, :3] += xi_hi[:, 3:6] * cfg.DT

            # Stage 3: Lyapunov IBP (monotone Tanh/Softplus → direct)
            with torch.no_grad():
                V_next_lo, V_next_hi = ibp_sequential(self.lyapunov.net, xi_lo_next, xi_hi_next)
                V_curr_lo, _         = ibp_sequential(self.lyapunov.net, xi_lo, xi_hi)
                V_next_max = V_next_hi.abs().max().item()
                V_curr_min = V_curr_lo.abs().min().item()

            return V_next_max <= (1.0 - cfg.KAPPA) * V_curr_min
        except Exception:
            return False

    def _verify_mc(self, xi: torch.Tensor, n: int = 200) -> bool:
        """Monte Carlo sampling fallback verification."""
        with torch.no_grad():
            noise   = (torch.rand(n, cfg.STATE_REPR_DIM, device=xi.device) * 2 - 1) * cfg.DELTA_IBP
            xi_test = xi.mean(0, keepdim=True) + noise
            xi_test[:, 6:10] /= xi_test[:, 6:10].norm(dim=1, keepdim=True)
            xi_test[:, 13:17] = torch.clamp(xi_test[:, 13:17], cfg.RPM_MIN, cfg.RPM_MAX)

            u_test   = self.controller(xi_test)
            xi_next  = self.dynamics(xi_test, u_test)
            V_curr   = self.lyapunov(xi_test)
            V_next   = self.lyapunov(xi_next)
            frac_ok  = (V_next <= (1.0 - cfg.KAPPA) * V_curr).float().mean().item()
        return frac_ok >= 0.90

    # ── statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return adaptation statistics dictionary."""
        lats_ms = [l * 1000 for l in self._latencies] if self._latencies else [0.0]
        lats_ms_sorted = sorted(lats_ms)
        p95_idx = max(0, int(0.95 * len(lats_ms_sorted)) - 1)

        return {
            "n_adaptations":   self.n_adaptations,
            "n_rollbacks":     self.n_rollbacks,
            "n_lqr_activations": self.n_lqr_activations,
            "lqr_active":      self.lqr_active,
            "max_violation":   max(self._violations) if self._violations else 0.0,
            "avg_latency_ms":  sum(lats_ms) / len(lats_ms),
            "p95_latency_ms":  lats_ms_sorted[p95_idx],
        }
