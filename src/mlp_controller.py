"""
Phase 2 — MLP Controller.

Architecture: 17 → 8 → 16 → 16 → 4  (LeakyReLU, α=0.01)

Yang24 Eq.2 equilibrium subtraction ensures π(ξ*) = u* for ANY weight θ:
    π_θ(ξ) = clamp( φ(ξ) − φ(ξ*) + u*,  RPM_MIN, RPM_MAX )
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import config as cfg


def _build_mlp(arch: list[int], activation: nn.Module) -> nn.Sequential:
    layers = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:          # no activation after last linear
            layers.append(activation)
    return nn.Sequential(*layers)


class MLPController(nn.Module):
    """
    MLP controller with Yang24 equilibrium-subtraction guarantee.

    Parameters
    ----------
    arch : list of ints
        Layer widths including input (17) and output (4).
    """

    def __init__(self, arch: list[int] = cfg.CONTROLLER_ARCH):
        super().__init__()
        self.net = _build_mlp(arch, nn.LeakyReLU(negative_slope=0.01))

        # Registered buffers move automatically with .to(device)
        self.register_buffer("xi_eq", cfg.EQUILIBRIUM_STATE.clone())
        self.register_buffer("u_eq",  cfg.EQUILIBRIUM_ACTION.clone())

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xi : [B, 17]  state tensor

        Returns:
            u  : [B,  4]  motor RPM commands in [RPM_MIN, RPM_MAX]
        """
        phi_xi = self.net(xi)
        phi_eq = self.net(self.xi_eq.unsqueeze(0).expand(xi.shape[0], -1))
        u_raw  = phi_xi - phi_eq + self.u_eq.unsqueeze(0)
        return torch.clamp(u_raw, cfg.RPM_MIN, cfg.RPM_MAX)
