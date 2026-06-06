"""
Phase 1 — Quadrotor Dynamics Model.

Implements the full 17D rigid-body quadrotor physics with:
  - WGS84 gravity model
  - ISA air-density model
  - X-configuration motor thrust/torque mapping
  - Euler's rotation equation
  - Quaternion kinematics with SO(3) re-normalisation
  - First-order motor lag
  - Linear + quadratic aerodynamic drag
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from src import config as cfg


class QuadrotorDynamics(nn.Module):
    """
    Batched quadrotor dynamics: xi_{t+1} = f(xi_t, u_t).

    State vector xi in R^17:
        [0:3]   p      – position (ENU, m)
        [3:6]   v      – velocity (ENU, m/s)
        [6:10]  q      – quaternion (w, x, y, z) on S³
        [10:13] omega  – body angular velocity (rad/s)
        [13:17] Omega  – motor RPMs

    Control u in R^4: motor RPM commands (clamped to [RPM_MIN, RPM_MAX]).
    """

    def __init__(self, dt: float = cfg.DT):
        super().__init__()
        self.dt = dt
        self.mass = cfg.DRONE_MASS
        self.J = torch.diag(torch.tensor(cfg.INERTIA_DIAG, dtype=cfg.DTYPE))
        self.J_inv = torch.inverse(self.J)
        self.k_f = cfg.K_F
        self.k_m = cfg.K_M
        self.arm = cfg.ARM_LENGTH / math.sqrt(2.0)
        self.motor_tau = cfg.MOTOR_TAU
        self.drag_lin = cfg.DRAG_LINEAR
        self.drag_quad = cfg.DRAG_QUADRATIC

        # gravity (WGS84 at Boston, sea level)
        self.g = cfg.GRAVITY_SEA_LEVEL

    # ── helpers ────────────────────────────────────────────────────────────────

    def motor_forces_torques(self, omega_rpm: torch.Tensor):
        """
        Given motor RPMs [B, 4], return (F_z, tau) where
          F_z  – total upward thrust  [B]
          tau  – body torques [B, 3] (roll, pitch, yaw)
        """
        omega_rad = omega_rpm * (2.0 * math.pi / 60.0)    # RPM → rad/s
        T = self.k_f * omega_rad ** 2                      # thrust per motor [B,4]

        F_z   = T.sum(dim=1)                               # total thrust [B]

        arm   = self.arm
        tau_r = arm  * ( T[:, 0] - T[:, 1] - T[:, 2] + T[:, 3])  # roll
        tau_p = arm  * (-T[:, 0] - T[:, 1] + T[:, 2] + T[:, 3])  # pitch
        tau_y = self.k_m * (-omega_rad[:, 0]**2 + omega_rad[:, 1]**2
                            -omega_rad[:, 2]**2 + omega_rad[:, 3]**2)  # yaw
        tau = torch.stack([tau_r, tau_p, tau_y], dim=1)   # [B, 3]
        return F_z, tau

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v [B,3] by quaternion q [B,4] = (w,x,y,z)."""
        w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]

        # Rodrigues / Hamilton product
        rx = (w**2 + x**2 - y**2 - z**2)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz
        ry = 2*(x*y + w*z)*vx + (w**2 - x**2 + y**2 - z**2)*vy + 2*(y*z - w*x)*vz
        rz = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (w**2 - x**2 - y**2 + z**2)*vz

        return torch.cat([rx, ry, rz], dim=1)

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, xi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        One Euler integration step.

        Args:
            xi : [B, 17]  current state
            u  : [B,  4]  motor RPM commands

        Returns:
            xi_next : [B, 17]  next state
        """
        dt = self.dt
        B  = xi.shape[0]

        pos   = xi[:, 0:3]    # [B, 3]
        vel   = xi[:, 3:6]
        q     = xi[:, 6:10]   # (w, x, y, z)
        omega = xi[:, 10:13]  # body frame
        rpms  = xi[:, 13:17]

        # ── Motor dynamics (first-order lag) ──────────────────────────────────
        u_cmd     = torch.clamp(u, cfg.RPM_MIN, cfg.RPM_MAX)
        rpms_next = rpms + dt / self.motor_tau * (u_cmd - rpms)

        # ── Thrust & torques ──────────────────────────────────────────────────
        F_z, tau  = self.motor_forces_torques(rpms)

        # ── Translational dynamics (ENU) ──────────────────────────────────────
        # Gravity in ENU: [0, 0, -g]
        g_enu     = torch.zeros(B, 3, dtype=xi.dtype, device=xi.device)
        g_enu[:, 2] = -self.g

        # Thrust vector in ENU: rotate body +z by q
        thrust_body = torch.zeros(B, 3, dtype=xi.dtype, device=xi.device)
        thrust_body[:, 2] = F_z / self.mass

        thrust_enu  = self._quat_rotate(q, thrust_body)

        # Aerodynamic drag (in ENU)
        vel_mag     = torch.norm(vel, dim=1, keepdim=True)
        drag_acc    = -(self.drag_lin + self.drag_quad * vel_mag) * vel / self.mass

        acc         = thrust_enu + g_enu + drag_acc

        vel_next    = vel  + dt * acc
        pos_next    = pos  + dt * vel

        # ── Rotational dynamics ───────────────────────────────────────────────
        J_inv = self.J_inv.to(xi.device)

        # omega x (J omega)
        Jw          = omega @ self.J.to(xi.device).t()
        cross       = torch.cross(omega, Jw, dim=1)
        alpha       = (tau - cross) @ J_inv.t()

        omega_next  = omega + dt * alpha

        # ── Quaternion kinematics  q_dot = 0.5 * Omega(omega) * q ────────────
        wx, wy, wz   = omega[:, 0], omega[:, 1], omega[:, 2]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        dqw = 0.5 * (-wx*qx - wy*qy - wz*qz)
        dqx = 0.5 * ( wx*qw + wz*qy - wy*qz)
        dqy = 0.5 * ( wy*qw - wz*qx + wx*qz)
        dqz = 0.5 * ( wz*qw + wy*qx - wx*qy)

        dq         = torch.stack([dqw, dqx, dqy, dqz], dim=1)
        q_next     = q + dt * dq
        # SO(3) re-normalisation — critical to prevent quaternion drift
        q_next     = q_next / torch.norm(q_next, dim=1, keepdim=True)

        xi_next = torch.cat([pos_next, vel_next, q_next, omega_next, rpms_next], dim=1)
        return xi_next
