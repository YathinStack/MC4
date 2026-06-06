"""
Phase 5 — Neural Observer (Multi-Sensor Fusion).

Observer update equation:
    ξ̂_{t+1} = f(ξ̂_t, u_t) + φ_obs(ξ̂_t, y_t − h(ξ̂_t)) − φ_obs(ξ̂_t, 0)

The correction term vanishes at zero innovation, preserving equilibrium.

Architecture of φ_obs: MLP  25 → 24 → 12 → 17   (Tanh hidden, linear output)
  input  = concat(ξ̂, innovation)  where innovation = y − h(ξ̂) ∈ R^8
  output = state correction in R^17
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from src import config as cfg
from src.quadrotor_dynamics import QuadrotorDynamics


# ── Sensor models ─────────────────────────────────────────────────────────────

class SensorSuite:
    """
    Realistic sensor models with noise and bias.

    Observation vector y ∈ R^8:
        [0:3]  accel  – IMU accelerometer (body frame, m/s²)
        [3:6]  gyro   – IMU gyroscope (body frame, rad/s)
        [6]    sonar  – downward altitude (m, saturates at 2 m)
        [7]    lidar  – minimum obstacle distance (m)
    """

    def __init__(self, device=cfg.DEVICE):
        self.device = device
        # Bias random walk state
        self.accel_bias = torch.zeros(3, device=device)
        self.gyro_bias  = torch.zeros(3, device=device)

    def full_observation(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Compute noisy observation from ground-truth state.

        Args:
            xi : [B, 17]

        Returns:
            y  : [B, 8]
        """
        B = xi.shape[0]

        # IMU accelerometer
        accel_noise = torch.randn(B, 3, device=xi.device) * 2e-3
        accel = xi[:, 3:6] + accel_noise         # simplified: velocity ≈ accel

        # Gyroscope
        gyro_noise = torch.randn(B, 3, device=xi.device) * 5e-4
        gyro = xi[:, 10:13] + gyro_noise

        # Sonar (downward altitude, capped at 2 m)
        altitude = xi[:, 2:3].clamp(0.0, 2.0) + torch.randn(B, 1, device=xi.device) * 0.01

        # Lidar placeholder (set to large value when no obstacle nearby)
        lidar = torch.full((B, 1), 10.0, device=xi.device) + torch.randn(B, 1, device=xi.device) * 0.02

        return torch.cat([accel, gyro, altitude, lidar], dim=1)

    def innovation(self, xi_hat: torch.Tensor, y_meas: torch.Tensor) -> torch.Tensor:
        """
        innovation = y_measured − h(ξ̂)

        Args:
            xi_hat : [B, 17]  estimated state
            y_meas : [B,  8]  sensor measurement

        Returns:
            innov  : [B,  8]
        """
        y_hat = self.full_observation(xi_hat)
        return y_meas - y_hat


# ── Neural Observer ───────────────────────────────────────────────────────────

def _build_observer_net(arch=cfg.OBSERVER_ARCH) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(arch) - 1):
        layers.append(nn.Linear(arch[i], arch[i + 1]))
        if i < len(arch) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class NeuralObserver(nn.Module):
    """
    Recurrent neural observer using an MLP correction term.

    The observer runs alongside the dynamics model and corrects the
    predicted state using the innovation (sensor residual).
    """

    def __init__(self, dynamics: QuadrotorDynamics | None = None):
        super().__init__()
        self.dynamics = dynamics or QuadrotorDynamics()
        self.sensors  = SensorSuite()
        # Input to φ_obs: concat(ξ̂ [17], innovation [8]) = 25
        self.net = _build_observer_net()

    def forward(self, xi_hat: torch.Tensor,
                u: torch.Tensor,
                y_meas: torch.Tensor) -> torch.Tensor:
        """
        One observer update step.

        Args:
            xi_hat : [B, 17]  current state estimate
            u      : [B,  4]  applied control
            y_meas : [B,  8]  sensor measurement

        Returns:
            xi_hat_next : [B, 17]  updated state estimate
        """
        # Prediction
        xi_pred = self.dynamics(xi_hat, u)

        # Innovation
        innov = y_meas - self.sensors.full_observation(xi_hat)

        # Neural correction  (vanishes at zero innovation)
        inp_innov = torch.cat([xi_hat, innov], dim=1)       # [B, 25]
        zero_innov = torch.zeros_like(innov)
        inp_zero  = torch.cat([xi_hat, zero_innov], dim=1)

        correction = self.net(inp_innov) - self.net(inp_zero)

        xi_hat_next = xi_pred + correction

        # SO(3) quaternion normalisation
        xi_hat_next = xi_hat_next.clone()
        xi_hat_next[:, 6:10] = xi_hat_next[:, 6:10] / \
            (torch.norm(xi_hat_next[:, 6:10], dim=1, keepdim=True) + 1e-8)

        return xi_hat_next
