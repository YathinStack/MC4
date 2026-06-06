"""
Phase 2 — Neural Lyapunov Certificate.

Two-term Lyapunov function (Yang24 Eq.3):
    V(ξ) = ‖φ_V(ξ) − φ_V(ξ*)‖₁  +  ‖(εI + R⊤R)(ξ − ξ*)‖₁

Properties guaranteed by construction:
    V(ξ*) = 0
    V(ξ)  > 0  ∀ ξ ≠ ξ*
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as cfg


class LyapunovNet(nn.Module):
    """
    Neural Lyapunov function with SVD-parameterised positive-definite R.

    Architecture of internal MLP φ_V: 17 → 32 → 16 → 1
      Hidden: Tanh
      Output: Softplus (strictly positive scalar)
    """

    def __init__(self, arch: list[int] = cfg.LYAPUNOV_ARCH):
        super().__init__()

        # Internal MLP φ_V
        layers: list[nn.Module] = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

        # SVD-parameterised R ensures R is full-rank ⟹ R⊤R is positive definite
        n = cfg.STATE_REPR_DIM
        U_init, _, Vh_init = torch.linalg.svd(torch.randn(n, n))
        self.register_buffer("U",  U_init)
        self.register_buffer("Vh", Vh_init)
        self.sigma_raw = nn.Parameter(torch.zeros(n))   # trainable singular values
        self.psi       = nn.Parameter(torch.zeros(n))   # extra positivity term

        # Equilibrium state (registered buffer → moves with device)
        self.register_buffer("xi_eq", cfg.EQUILIBRIUM_STATE.clone())

    @property
    def R(self) -> torch.Tensor:
        """R = U · diag(softplus(σ) + ψ²) · Vhᵀ  — always full rank."""
        sv = F.softplus(self.sigma_raw) + self.psi ** 2
        return self.U @ torch.diag(sv) @ self.Vh

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi : [B, 17]

        Returns:
            V  : [B]  Lyapunov values (scalar per sample, > 0 for xi ≠ xi*)
        """
        xi_eq  = self.xi_eq.unsqueeze(0).to(xi.device)

        # Term 1 – neural component
        phi_xi = self.net(xi)
        phi_eq = self.net(xi_eq.expand(xi.shape[0], -1))
        V_nn   = torch.abs(phi_xi - phi_eq).squeeze(1)

        # Term 2 – quadratic-1-norm component
        delta  = xi - xi_eq
        M      = cfg.EPSILON_PD * torch.eye(delta.shape[1],
                                            dtype=xi.dtype, device=xi.device) \
                 + self.R.t() @ self.R
        Mdelta = F.linear(delta, M)
        V_lin  = torch.norm(Mdelta, p=1, dim=1)

        return V_nn + V_lin
