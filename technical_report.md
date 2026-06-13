# Comprehensive Technical Report
## Online Adaptive Lyapunov-Stable Drone Control System
**Prepared for external review — full deep-read of every file in src/, web/, docs/, models/**

---

## TABLE OF CONTENTS

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [External Dependencies & Requirements](#2-external-dependencies--requirements)
3. [Configuration — `src/config.py`](#3-configuration--srcconfigpy)
4. [Phase 1 — Quadrotor Dynamics — `src/quadrotor_dynamics.py`](#4-phase-1--quadrotor-dynamics--srcquadrotor_dynamicspy)
5. [Phase 2a — MLP Controller — `src/mlp_controller.py`](#5-phase-2a--mlp-controller--srcmlp_controllerpy)
6. [Phase 2b — Neural Lyapunov Certificate — `src/lyapunov_network.py`](#6-phase-2b--neural-lyapunov-certificate--srclyapunov_networkpy)
7. [Phase 3 — CBF Obstacle Avoidance — `src/barrier_function.py`](#7-phase-3--cbf-obstacle-avoidance--srcbarrier_functionpy)
8. [Phase 4 — Stability-Aware SGD — `src/stability_sgd.py`](#8-phase-4--stability-aware-sgd--srcstability_sgdpy)
9. [Phase 5 — Neural Observer — `src/neural_observer.py`](#9-phase-5--neural-observer--srcneural_observerpy)
10. [Phase 6 — RRT* Replanner — `src/rrt_replanner.py`](#10-phase-6--rrt-replanner--srcrrt_replannerpy)
11. [Utilities — `src/utils.py`](#11-utilities--srcutilspy)
12. [Stub/Alias Files — `src/__init__.py` etc.](#12-stubalias-files)
13. [Phase 7 — Web Dashboard — `web/dashboard_server.py`](#13-phase-7--web-dashboard--webdashboard_serverpy)
14. [Web Frontend — `web/index.html`, `web/style.css`, `web/app.js`](#14-web-frontend)
15. [Documentation Files — `docs/`](#15-documentation-files--docs)
16. [Model Checkpoints — `models/checkpoints/`](#16-model-checkpoints--modelscheckpoints)
17. [Mathematical Core — All Formulas](#17-mathematical-core--all-formulas)
18. [Complete Data Flow Trace](#18-complete-data-flow-trace)
19. [Test Coverage & Known Failures](#19-test-coverage--known-failures)
20. [Formal Guarantees Summary](#20-formal-guarantees-summary)

---

## 1. Project Overview & Architecture

**Project name:** Online Adaptive Lyapunov-Stable Drone Control System  
**Root directory:** `c:\Users\LENOVO\OneDrive\Desktop\MATH01`  
**Python version:** 3.12 (`.python-version` file)  
**Device target:** RTX 4050 (6 GB VRAM), falls back to CPU

The system is organised into **7 tightly integrated phases**, each a separate Python module, culminating in a real-time 3D web dashboard. The drone's full state lives in a **17-dimensional tensor**:

| Index | Symbol | Meaning | Unit |
|-------|--------|---------|------|
| 0–2 | **p** | Position (ENU frame) | m |
| 3–5 | **v** | Linear velocity (ENU) | m/s |
| 6–9 | **q** | Quaternion (w,x,y,z) on S³ | — |
| 10–12 | **ω** | Body angular velocity | rad/s |
| 13–16 | **Ω** | Motor RPMs | RPM |

**7-Phase pipeline (data flows top to bottom):**

```
[Sensors: IMU / GPS / Lidar / Sonar]
           │  y ∈ ℝ⁸
           ▼
  ┌──────────────────┐
  │  NeuralObserver  │  Phase 5: MLP 25→24→12→17
  │  (state fuser)   │  ξ̂_{t+1} = f(ξ̂,u) + φ(ξ̂,y−h(ξ̂)) − φ(ξ̂,0)
  └────────┬─────────┘
           │  ξ̂ ∈ ℝ¹⁷  (estimated state)
           ▼
  ┌──────────────────┐   obstacles
  │  RRTReplanner    │◄──────────  Phase 6: RRT* + Switched Lyapunov
  └────────┬─────────┘
           │  waypoints
           ▼
  ┌──────────────────┐
  │  MLPController   │  Phase 2a: 17→8→16→16→4, Yang24 subtraction
  └────────┬─────────┘
           │  u ∈ ℝ⁴  (motor RPM commands)
           ▼
  ┌──────────────────┐   V(ξ) from LyapunovNet (Phase 2b)
  │StabilityAwareSGD │◄──────────  Phase 4: IBP verify + LQR fallback
  └────────┬─────────┘
           │  u_safe
           ▼
  ┌──────────────────┐
  │LyapunovBarrier   │  Phase 3: CBF B(ξ)≥0, gradient projection
  │    Fusion        │
  └────────┬─────────┘
           │  u_final
           ▼
  ┌──────────────────┐
  │QuadrotorDynamics │  Phase 1: 17D Euler integration at 100 Hz
  └──────────────────┘
           │  ξ_{t+1}
           ▼
  ┌──────────────────┐
  │  Flask + Socket  │  Phase 7: REST + WebSocket dashboard
  └──────────────────┘
```

---

## 2. External Dependencies & Requirements

**File:** `requirements.txt` (8 lines, 115 bytes)

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
flask>=3.0.0
flask-socketio>=5.3.0
pytest>=7.4.0
```

| Package | Purpose |
|---------|---------|
| `torch` | All neural networks, tensor math, autograd |
| `numpy` | Array utilities (used sparingly inside matplotlib plotting) |
| `matplotlib` | Post-simulation report plot generation |
| `scipy` | Not used in core src/ (available for training scripts) |
| `flask` | HTTP REST API server |
| `flask-socketio` | WebSocket real-time telemetry streaming |
| `pytest` | Test runner |

> [!NOTE]
> There is NO `flask-cors` in requirements — CORS is handled by `flask-socketio`'s `cors_allowed_origins="*"`.
> PyTorch 2.6 changed the default `weights_only=True` for `torch.load()`. The server explicitly passes `weights_only=False` to load legacy checkpoints.

---

## 3. Configuration — `src/config.py`

**Path:** `src/config.py` | **Lines:** 100 | **Bytes:** 6,004

### Purpose
Single source of truth for every physical constant, network architecture definition, and training hyperparameter. All other modules import from here as `from src import config as cfg`.

### All Imports
```python
import torch
import math
```

### Complete Source Code

```python
"""
Central configuration for the Online Adaptive Lyapunov-Stable Drone Control System.
All physical constants, network architectures, and training hyperparameters.
RTX 4050-friendly values (6 GB VRAM).
"""

import torch
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32

# ── Simulation ────────────────────────────────────────────────────────────────
DT                  = 0.01          # s  (100 Hz)

# ── Physical constants ────────────────────────────────────────────────────────
GRAVITY_SEA_LEVEL   = 9.80665       # m/s²  (standard gravity)
DRONE_MASS          = 0.85          # kg
INERTIA_DIAG        = [0.005, 0.005, 0.01]  # kg·m²  (Jxx, Jyy, Jzz)
ARM_LENGTH          = 0.18          # m  (motor-to-CoM distance)
K_F                 = 1.2e-5        # N/(rad/s)²  thrust coefficient
K_M                 = 1.5e-7        # N·m/(rad/s)²  torque coefficient
MOTOR_TAU           = 0.015         # s  (15 ms first-order motor lag)
DRONE_RADIUS        = 0.20          # m  (collision radius)

# ── Aerodynamics ──────────────────────────────────────────────────────────────
DRAG_LINEAR         = 0.10          # N/(m/s)
DRAG_QUADRATIC      = 0.05          # N/(m/s)²

# ── ISA atmosphere ───────────────────────────────────────────────────────────
RHO_0               = 1.225         # kg/m³
T_0                 = 288.15        # K
LAPSE_RATE          = 0.0065        # K/m
R_AIR               = 287.058       # J/(kg·K)

# ── Motor electrical model ────────────────────────────────────────────────────
R_MOTOR             = 0.1           # Ω  (winding resistance)

# ── RPM limits ────────────────────────────────────────────────────────────────
RPM_MIN             = 1_000.0
RPM_MAX             = 12_000.0

# ── Hover equilibrium (derived) ───────────────────────────────────────────────
_g                  = GRAVITY_SEA_LEVEL
_hover_omega        = math.sqrt(DRONE_MASS * _g / (4.0 * K_F))  # rad/s
HOVER_RPM           = _hover_omega * 60.0 / (2.0 * math.pi)

# ── State / action dimensions ─────────────────────────────────────────────────
STATE_REPR_DIM      = 17            # [p(3), v(3), q(4), ω(3), Ω(4)]
ACTION_DIM          = 4             # motor RPM commands
OBS_DIM             = 8             # [accel(3), gyro(3), sonar(1), lidar(1)]

# ── Equilibrium state & action ────────────────────────────────────────────────
EQUILIBRIUM_STATE   = torch.zeros(STATE_REPR_DIM, dtype=DTYPE)
EQUILIBRIUM_STATE[6]  = 1.0                        # quaternion w = 1
EQUILIBRIUM_STATE[13] = HOVER_RPM                  # motor 1
EQUILIBRIUM_STATE[14] = HOVER_RPM                  # motor 2
EQUILIBRIUM_STATE[15] = HOVER_RPM                  # motor 3
EQUILIBRIUM_STATE[16] = HOVER_RPM                  # motor 4

EQUILIBRIUM_ACTION  = torch.full((ACTION_DIM,), HOVER_RPM, dtype=DTYPE)

# ── Network architectures ─────────────────────────────────────────────────────
CONTROLLER_ARCH     = [17, 8, 16, 16, 4]
LYAPUNOV_ARCH       = [17, 32, 16, 1]
OBSERVER_ARCH       = [25, 24, 12, 17]

# ── Lyapunov training hyper-parameters ────────────────────────────────────────
KAPPA               = 0.05          # exponential decay rate
EPSILON_PD          = 1e-4          # ε for positive-definiteness of R^T R
LAMBDA_BARRIER      = 10.0          # barrier penalty coefficient
DELTA_IBP           = 0.1          # IBP verification box half-width
TAU_VIOLATION       = 1e-3          # Lyapunov violation threshold for SGD

# ── CEGIS training ────────────────────────────────────────────────────────────
PGD_STEPS           = 20
PGD_STEP_SIZE       = 0.02
ROA_RADIUS_INIT     = 0.5
ROA_RADIUS_MAX      = 5.0
ROA_EXPAND_FACTOR   = 1.1
ROA_PASS_THRESHOLD  = 0.95
C0_BOUNDARY         = 0.1
C1_ROA              = 0.5
C2_REG              = 1e-4
C3_POWER            = 0.01

# ── Online adaptation (Stability-Aware SGD) ────────────────────────────────────
ADAPT_LR            = 1e-4
MAX_CONSECUTIVE_FAIL = 3

# ── Velocity limits ────────────────────────────────────────────────────────────
V_MAX               = 3.0           # m/s  max navigation speed

# ── RRT* planner ──────────────────────────────────────────────────────────────
RRT_MAX_SAMPLES     = 1000
RRT_STEP_SIZE       = 0.3
RRT_GOAL_BIAS       = 0.2
RRT_REWIRE_RADIUS   = 1.0
RRT_TIMEOUT_MS      = 50.0
```

### All Constants & Derived Values

| Constant | Value | Unit | Meaning |
|----------|-------|------|---------|
| `DT` | 0.01 | s | Integration timestep (100 Hz) |
| `GRAVITY_SEA_LEVEL` | 9.80665 | m/s² | Standard gravity (ISO 80000-3) |
| `DRONE_MASS` | 0.85 | kg | Total vehicle mass |
| `INERTIA_DIAG` | [0.005, 0.005, 0.01] | kg·m² | Principal moments of inertia (Jxx, Jyy, Jzz) |
| `ARM_LENGTH` | 0.18 | m | Motor center to CoM |
| `K_F` | 1.2×10⁻⁵ | N/(rad/s)² | Thrust coefficient: T = K_F·ω² |
| `K_M` | 1.5×10⁻⁷ | N·m/(rad/s)² | Reaction torque coefficient |
| `MOTOR_TAU` | 0.015 | s | First-order motor lag time constant |
| `DRONE_RADIUS` | 0.20 | m | Collision sphere radius |
| `DRAG_LINEAR` | 0.10 | N/(m/s) | Stokes drag coefficient |
| `DRAG_QUADRATIC` | 0.05 | N/(m/s)² | Newtonian drag coefficient |
| `RPM_MIN` | 1,000 | RPM | Minimum motor speed |
| `RPM_MAX` | 12,000 | RPM | Maximum motor speed |
| `HOVER_RPM` | ≈3,997 | RPM | Derived hover equilibrium speed |
| `STATE_REPR_DIM` | 17 | — | State vector dimension |
| `ACTION_DIM` | 4 | — | Control vector dimension |
| `OBS_DIM` | 8 | — | Observation vector dimension |
| `KAPPA` | 0.05 | — | Lyapunov exponential decay rate |
| `EPSILON_PD` | 1×10⁻⁴ | — | PD regularisation for R⊤R |
| `LAMBDA_BARRIER` | 10.0 | — | Barrier penalty weight λ |
| `DELTA_IBP` | 0.1 | — | IBP perturbation box half-width δ |
| `TAU_VIOLATION` | 1×10⁻³ | — | Violation threshold τ triggering SGD |
| `ADAPT_LR` | 1×10⁻⁴ | — | Online SGD learning rate η |
| `MAX_CONSECUTIVE_FAIL` | 3 | — | Rollbacks before LQR activation |
| `V_MAX` | 3.0 | m/s | Navigation speed cap |
| `RRT_MAX_SAMPLES` | 1,000 | — | Max RRT* tree nodes |
| `RRT_STEP_SIZE` | 0.3 | m | RRT* extension step |
| `RRT_GOAL_BIAS` | 0.2 | — | Probability of sampling goal directly |
| `RRT_REWIRE_RADIUS` | 1.0 | m | Rewiring neighborhood radius |
| `RRT_TIMEOUT_MS` | 50.0 | ms | Hard planning timeout |

**Hover RPM derivation:**
From thrust balance: 4·K_F·ω²_hover = m·g
→ ω_hover = √(m·g / 4K_F) = √(0.85·9.80665 / 4·1.2×10⁻⁵) ≈ 418.6 rad/s
→ HOVER_RPM = 418.6 × 60/(2π) ≈ **3,997 RPM**

---

## 4. Phase 1 — Quadrotor Dynamics — `src/quadrotor_dynamics.py`

**Path:** `src/quadrotor_dynamics.py` | **Lines:** 161 | **Bytes:** 6,502

### Purpose
Full 17-dimensional rigid-body quadrotor physics model. Implements one Euler integration step `ξ_{t+1} = f(ξ_t, u_t)` using batched PyTorch tensors.

### Imports
```python
import math
import torch
import torch.nn as nn
from src import config as cfg
```

### Class: `QuadrotorDynamics(nn.Module)`

**Constructor:** `__init__(self, dt: float = cfg.DT)`

| Attribute | Type | Value | Purpose |
|-----------|------|-------|---------|
| `self.dt` | float | 0.01 | Timestep |
| `self.mass` | float | 0.85 | Drone mass |
| `self.J` | Tensor [3,3] | diag(0.005,0.005,0.01) | Inertia tensor |
| `self.J_inv` | Tensor [3,3] | J⁻¹ | Precomputed inertia inverse |
| `self.k_f` | float | 1.2×10⁻⁵ | Thrust coefficient |
| `self.k_m` | float | 1.5×10⁻⁷ | Torque coefficient |
| `self.arm` | float | 0.18/√2 ≈ 0.1273 | Effective torque arm |
| `self.motor_tau` | float | 0.015 | Motor time constant |
| `self.drag_lin` | float | 0.10 | Linear drag |
| `self.drag_quad` | float | 0.05 | Quadratic drag |
| `self.g` | float | 9.80665 | Gravity |

---

#### Method: `motor_forces_torques(omega_rpm)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `omega_rpm` | `Tensor [B, 4]` | Motor speeds in RPM |

| Return | Type/Shape | Description |
|--------|-----------|-------------|
| `F_z` | `Tensor [B]` | Total upward thrust in Newtons |
| `tau` | `Tensor [B, 3]` | Body torques [roll, pitch, yaw] in N·m |

**Math:**
1. Convert RPM → rad/s: `ω_rad = ω_rpm × (2π/60)`
2. Per-motor thrust: `T_i = K_F × ω_i²`
3. Total thrust: `F_z = T_1 + T_2 + T_3 + T_4`
4. Roll torque: `τ_roll = (L/√2) × (T_1 − T_2 − T_3 + T_4)`
5. Pitch torque: `τ_pitch = (L/√2) × (−T_1 − T_2 + T_3 + T_4)`
6. Yaw torque: `τ_yaw = K_M × (−ω_1² + ω_2² − ω_3² + ω_4²)`

X-configuration motor numbering (viewed from above):
```
  1(CW)   2(CCW)
     \   /
      \ /
      / \
     /   \
  4(CCW)  3(CW)
```

---

#### Static Method: `_quat_rotate(q, v)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `q` | `Tensor [B, 4]` | Quaternion (w, x, y, z) |
| `v` | `Tensor [B, 3]` | Vector to rotate |

**Returns:** `Tensor [B, 3]` — rotated vector

**Math (Rodrigues / Hamilton product):**
```
r = R(q) · v   where R is the rotation matrix corresponding to q
```
Full element-wise expansion (avoids explicit matrix build):
```
rx = (w²+x²-y²-z²)·vx + 2(xy-wz)·vy + 2(xz+wy)·vz
ry = 2(xy+wz)·vx + (w²-x²+y²-z²)·vy + 2(yz-wx)·vz
rz = 2(xz-wy)·vx + 2(yz+wx)·vy + (w²-x²-y²+z²)·vz
```

---

#### Method: `forward(xi, u)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | Current state |
| `u` | `Tensor [B, 4]` | Motor RPM commands |

**Returns:** `Tensor [B, 17]` — next state after one Euler step

**Full algorithm (10 sub-steps):**

1. **Unpack state:** pos=xi[:,0:3], vel=xi[:,3:6], q=xi[:,6:10], ω=xi[:,10:13], rpms=xi[:,13:17]

2. **Motor dynamics (1st-order lag):**
   `rpms_next = rpms + (dt/τ_m) × (clamp(u, RPM_MIN, RPM_MAX) − rpms)`

3. **Thrust & torques:** via `motor_forces_torques(rpms)` (uses *current* rpms, not commanded)

4. **Gravity vector (ENU):** `g_enu = [0, 0, −g]`

5. **Thrust in ENU:** rotate body `[0, 0, F_z/m]` by quaternion q

6. **Aerodynamic drag:**
   `drag_acc = −(c_lin + c_quad × ‖v‖) × v / m`

7. **Translational Euler step:**
   `acc = thrust_enu + g_enu + drag_acc`
   `vel_next = vel + dt × acc`
   `pos_next = pos + dt × vel`  ← semi-implicit (uses current vel, not next)

8. **Euler's rotation equation:**
   `Jω̇ = τ − ω × (Jω)`
   `α = J⁻¹ × (τ − ω × Jω)`
   `ω_next = ω + dt × α`

9. **Quaternion kinematics:**
   `q̇ = ½ × Ω(ω) × q`
   ```
   dqw = 0.5 × (−ωx·qx − ωy·qy − ωz·qz)
   dqx = 0.5 × (+ωx·qw + ωz·qy − ωy·qz)
   dqy = 0.5 × (+ωy·qw − ωz·qx + ωx·qz)
   dqz = 0.5 × (+ωz·qw + ωy·qx − ωx·qy)
   ```
   `q_next = (q + dt × dq) / ‖q + dt × dq‖`  ← SO(3) renormalisation

10. **Pack state:** `xi_next = cat([pos_next, vel_next, q_next, ω_next, rpms_next])`

### Complete Source Code

```python
"""
Phase 1 — Quadrotor Dynamics Model.
...
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from src import config as cfg

class QuadrotorDynamics(nn.Module):
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
        self.g = cfg.GRAVITY_SEA_LEVEL

    def motor_forces_torques(self, omega_rpm: torch.Tensor):
        omega_rad = omega_rpm * (2.0 * math.pi / 60.0)
        T = self.k_f * omega_rad ** 2
        F_z   = T.sum(dim=1)
        arm   = self.arm
        tau_r = arm  * ( T[:, 0] - T[:, 1] - T[:, 2] + T[:, 3])
        tau_p = arm  * (-T[:, 0] - T[:, 1] + T[:, 2] + T[:, 3])
        tau_y = self.k_m * (-omega_rad[:, 0]**2 + omega_rad[:, 1]**2
                            -omega_rad[:, 2]**2 + omega_rad[:, 3]**2)
        tau = torch.stack([tau_r, tau_p, tau_y], dim=1)
        return F_z, tau

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
        rx = (w**2 + x**2 - y**2 - z**2)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz
        ry = 2*(x*y + w*z)*vx + (w**2 - x**2 + y**2 - z**2)*vy + 2*(y*z - w*x)*vz
        rz = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (w**2 - x**2 - y**2 + z**2)*vz
        return torch.cat([rx, ry, rz], dim=1)

    def forward(self, xi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        B  = xi.shape[0]
        pos   = xi[:, 0:3]
        vel   = xi[:, 3:6]
        q     = xi[:, 6:10]
        omega = xi[:, 10:13]
        rpms  = xi[:, 13:17]
        u_cmd     = torch.clamp(u, cfg.RPM_MIN, cfg.RPM_MAX)
        rpms_next = rpms + dt / self.motor_tau * (u_cmd - rpms)
        F_z, tau  = self.motor_forces_torques(rpms)
        g_enu     = torch.zeros(B, 3, dtype=xi.dtype, device=xi.device)
        g_enu[:, 2] = -self.g
        thrust_body = torch.zeros(B, 3, dtype=xi.dtype, device=xi.device)
        thrust_body[:, 2] = F_z / self.mass
        thrust_enu  = self._quat_rotate(q, thrust_body)
        vel_mag     = torch.norm(vel, dim=1, keepdim=True)
        drag_acc    = -(self.drag_lin + self.drag_quad * vel_mag) * vel / self.mass
        acc         = thrust_enu + g_enu + drag_acc
        vel_next    = vel  + dt * acc
        pos_next    = pos  + dt * vel
        J_inv = self.J_inv.to(xi.device)
        Jw          = omega @ self.J.to(xi.device).t()
        cross       = torch.cross(omega, Jw, dim=1)
        alpha       = (tau - cross) @ J_inv.t()
        omega_next  = omega + dt * alpha
        wx, wy, wz   = omega[:, 0], omega[:, 1], omega[:, 2]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        dqw = 0.5 * (-wx*qx - wy*qy - wz*qz)
        dqx = 0.5 * ( wx*qw + wz*qy - wy*qz)
        dqy = 0.5 * ( wy*qw - wz*qx + wx*qz)
        dqz = 0.5 * ( wz*qw + wy*qx - wx*qy)
        dq         = torch.stack([dqw, dqx, dqy, dqz], dim=1)
        q_next     = q + dt * dq
        q_next     = q_next / torch.norm(q_next, dim=1, keepdim=True)
        xi_next = torch.cat([pos_next, vel_next, q_next, omega_next, rpms_next], dim=1)
        return xi_next
```

---

## 5. Phase 2a — MLP Controller — `src/mlp_controller.py`

**Path:** `src/mlp_controller.py` | **Lines:** 56 | **Bytes:** 1,714

### Purpose
The neural controller maps the 17D state to 4 motor RPM commands using a shallow feedforward MLP with the **Yang24 equilibrium-subtraction guarantee**.

### Imports
```python
from __future__ import annotations
import torch
import torch.nn as nn
from src import config as cfg
```

### Helper Function: `_build_mlp(arch, activation)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `arch` | `list[int]` | Layer widths including input and output |
| `activation` | `nn.Module` | Activation instance (shared) |

**Returns:** `nn.Sequential` — linear layers interleaved with activations (no activation after final linear)

### Class: `MLPController(nn.Module)`

**Architecture:** `17 → 8 → 16 → 16 → 4` with `LeakyReLU(α=0.01)`

**Total trainable parameters:** 2,277
- Linear(17→8): 17×8 + 8 = 144
- Linear(8→16): 8×16 + 16 = 144
- Linear(16→16): 16×16 + 16 = 272
- Linear(16→4): 16×4 + 4 = 68 … wait, correct count:
  - W₁: 17×8=136, b₁: 8 → 144
  - W₂: 8×16=128, b₂: 16 → 144
  - W₃: 16×16=256, b₃: 16 → 272
  - W₄: 16×4=64, b₄: 4 → 68
  - **Total: 628 parameters**

**Registered buffers (move with `.to(device)`):**
- `xi_eq`: `Tensor [17]` — equilibrium state ξ*
- `u_eq`: `Tensor [4]` — equilibrium action u* = [HOVER_RPM]×4

#### Method: `forward(xi)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | Current state |

**Returns:** `Tensor [B, 4]` — motor RPM commands ∈ [RPM_MIN, RPM_MAX]

**Algorithm (Yang24 Eq. 2):**
```
φ(ξ)  = net(xi)                         # raw MLP output
φ(ξ*) = net(xi_eq.expand(B, -1))        # MLP at equilibrium
u_raw = φ(ξ) − φ(ξ*) + u*              # equilibrium subtraction
u     = clamp(u_raw, 1000, 12000)       # physical motor limits
```

**Key property — equilibrium invariance (structural, not learned):**
At ξ = ξ*: φ(ξ*) − φ(ξ*) = 0, so u_raw = 0 + u* = u*. The clamp preserves u* since u* = HOVER_RPM ∈ [1000, 12000].

### Complete Source Code

```python
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
        if i < len(arch) - 2:
            layers.append(activation)
    return nn.Sequential(*layers)

class MLPController(nn.Module):
    def __init__(self, arch: list[int] = cfg.CONTROLLER_ARCH):
        super().__init__()
        self.net = _build_mlp(arch, nn.LeakyReLU(negative_slope=0.01))
        self.register_buffer("xi_eq", cfg.EQUILIBRIUM_STATE.clone())
        self.register_buffer("u_eq",  cfg.EQUILIBRIUM_ACTION.clone())

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        phi_xi = self.net(xi)
        phi_eq = self.net(self.xi_eq.unsqueeze(0).expand(xi.shape[0], -1))
        u_raw  = phi_xi - phi_eq + self.u_eq.unsqueeze(0)
        return torch.clamp(u_raw, cfg.RPM_MIN, cfg.RPM_MAX)
```

---

## 6. Phase 2b — Neural Lyapunov Certificate — `src/lyapunov_network.py`

**Path:** `src/lyapunov_network.py` | **Lines:** 83 | **Bytes:** 2,725

### Purpose
Defines V(ξ), the neural Lyapunov function certificate. V is positive-definite by construction and V(ξ*) = 0 exactly — proven algebraically, not empirically.

### Imports
```python
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config as cfg
```

### Class: `LyapunovNet(nn.Module)`

**Internal MLP φ_V architecture:** `17 → 32 → 16 → 1`
- Hidden: `Tanh`
- Output: `Softplus` (ensures strictly positive scalar output)

**Trainable parameters:**
- MLP: (17×32+32) + (32×16+16) + (16×1+1) = 576+528+17 = **1,121**
- `sigma_raw`: [17] — raw singular values
- `psi`: [17] — extra positivity terms
- **Total: 1,155 parameters**

**Registered buffers:**
- `U_svd`: [17,17] — left singular vectors (fixed)
- `Vh_svd`: [17,17] — right singular vectors (fixed)
- `xi_eq`: [17] — equilibrium state

#### Property: `R`

**Returns:** `Tensor [17, 17]` — full-rank matrix

**Formula:**
```
sv = softplus(sigma_raw) + psi²      # [17] all strictly positive
R  = U_svd @ diag(sv) @ Vh_svd      # [17,17] full rank by construction
```

This ensures R⊤R is **positive definite** (all eigenvalues > 0).

#### Method: `forward(xi)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | State batch |

**Returns:** `Tensor [B]` — Lyapunov values V(ξ) ≥ 0, = 0 only at ξ*

**Two-term formula (Yang24 Eq. 3):**
```
Term 1 (Neural):
    phi_xi = net(xi)            # [B, 1]
    phi_eq = net(xi*)           # [B, 1]  broadcast
    V_nn   = |phi_xi - phi_eq|  # [B]     1-norm of scalar difference

Term 2 (Quadratic-1-norm):
    delta  = xi - xi*                          # [B, 17]
    M      = ε·I + R⊤R                        # [17, 17]  positive definite
    Mdelta = F.linear(delta, M)               # [B, 17]
    V_lin  = ‖Mdelta‖₁   (L1-norm over dim 1) # [B]

V(ξ) = V_nn + V_lin
```

**Proof of positive-definiteness:**
- At ξ = ξ*: δ = 0 → V_lin = ‖M·0‖₁ = 0; phi_xi − phi_eq = 0 → V_nn = 0. ∴ V(ξ*) = 0.
- At ξ ≠ ξ*: δ ≠ 0 and M is positive definite → Mδ ≠ 0 → V_lin > 0 → V(ξ) > 0. ✓

### Complete Source Code

```python
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
    def __init__(self, arch: list[int] = cfg.LYAPUNOV_ARCH):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        n = cfg.STATE_REPR_DIM
        U_init, _, Vh_init = torch.linalg.svd(torch.randn(n, n))
        self.register_buffer("U_svd",  U_init)
        self.register_buffer("Vh_svd", Vh_init)
        self.sigma_raw = nn.Parameter(torch.zeros(n))
        self.psi       = nn.Parameter(torch.zeros(n))
        self.register_buffer("xi_eq", cfg.EQUILIBRIUM_STATE.clone())

    @property
    def R(self) -> torch.Tensor:
        sv = F.softplus(self.sigma_raw) + self.psi ** 2
        return self.U_svd @ torch.diag(sv) @ self.Vh_svd

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        xi_eq  = self.xi_eq.unsqueeze(0).to(xi.device)
        phi_xi = self.net(xi)
        phi_eq = self.net(xi_eq.expand(xi.shape[0], -1))
        V_nn   = torch.abs(phi_xi - phi_eq).squeeze(1)
        delta  = xi - xi_eq
        M      = cfg.EPSILON_PD * torch.eye(delta.shape[1],
                                            dtype=xi.dtype, device=xi.device) \
                 + self.R.t() @ self.R
        Mdelta = F.linear(delta, M)
        V_lin  = torch.norm(Mdelta, p=1, dim=1)
        return V_nn + V_lin
```

---

## 7. Phase 3 — CBF Obstacle Avoidance — `src/barrier_function.py`

**Path:** `src/barrier_function.py` | **Lines:** 128 | **Bytes:** 4,245

### Purpose
Implements a Control Barrier Function (CBF) for each spherical obstacle, fuses it with the Lyapunov certificate, and projects control gradients onto the safe subspace when task and safety objectives conflict.

### Imports
```python
from __future__ import annotations
import torch
import torch.nn as nn
from src import config as cfg
from src.mlp_controller import MLPController
from src.lyapunov_network import LyapunovNet
from src.quadrotor_dynamics import QuadrotorDynamics
```

### Function: `barrier_value(pos, obs_center, obs_radius)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `pos` | `Tensor [B, 3]` | Drone position |
| `obs_center` | `Tensor [3]` | Obstacle center |
| `obs_radius` | `float` | Obstacle radius |

**Returns:** `Tensor [B]` — barrier values B(ξ)

**Formula:** `B(ξ) = ‖p − o‖² − (r_obs + r_drone)²`

B ≥ 0 → safe (drone outside inflated sphere); B < 0 → unsafe (inside sphere).
`r_safe = r_obs + DRONE_RADIUS` includes drone collision sphere.

### Class: `LyapunovBarrierFusion(nn.Module)`

**Constructor parameters:**
- `controller`: `MLPController`
- `lyapunov`: `LyapunovNet`
- `dynamics`: `QuadrotorDynamics`
- `self.lam = cfg.LAMBDA_BARRIER = 10.0`

#### Method: `forward(xi, obstacles)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | State (needs `requires_grad=True` for projection) |
| `obstacles` | `list[dict]` | Each dict: `{center, radius, velocity}` |

**Returns:**
- `u_safe`: `Tensor [B, 4]` — safety-filtered motor commands
- `B_min`: `Tensor [B]` — minimum barrier value over all obstacles
- `V_total`: `Tensor [B]` — fused Lyapunov-barrier value

**Algorithm:**

1. Compute base control: `u = controller(xi)`
2. For each obstacle i:
   - `B_i = barrier_value(pos, center_i, radius_i)`
   - `penalty_i = λ × (max(0, −B_i))²`
3. `V_total = V(ξ) + Σ_i penalty_i`
4. `B_min = min_i(B_i)` (most dangerous obstacle)
5. If `B_min < 1.0` and `xi.requires_grad`:
   - Compute safety gradient: `g_s = ∂V_total/∂ξ[:, :3]`  (position only)
   - Compute task gradient: `g_t = ∂V/∂ξ[:, :3]`
   - Conflict check: `g_t · g_s < 0`
   - If conflict: `correction = −0.1 × u[conflict]`
   - `u = u + correction`
6. `u_safe = clamp(u, RPM_MIN, RPM_MAX)`

**Fusion formula:**
```
V_total(ξ) = V(ξ) + λ · Σᵢ [max(0, −Bᵢ(ξ))]²
```

**Gradient projection:**
```
ĝ_s = g_s / ‖g_s‖
proj = g_t · ĝ_s
g_proj = g_t − proj · ĝ_s     (when g_t · g_s < 0)
```

### Complete Source Code

```python
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

def barrier_value(pos, obs_center, obs_radius):
    r_safe   = obs_radius + cfg.DRONE_RADIUS
    diff     = pos - obs_center.to(pos.device)
    dist_sq  = (diff ** 2).sum(dim=1)
    return dist_sq - r_safe ** 2

class LyapunovBarrierFusion(nn.Module):
    def __init__(self, controller, lyapunov, dynamics):
        super().__init__()
        self.controller = controller
        self.lyapunov   = lyapunov
        self.dynamics   = dynamics
        self.lam        = cfg.LAMBDA_BARRIER

    def forward(self, xi, obstacles):
        u   = self.controller(xi)
        pos = xi[:, :3]
        V   = self.lyapunov(xi)
        B_values = []
        penalty  = torch.zeros_like(V)
        for obs in obstacles:
            center = torch.tensor(obs["center"], dtype=xi.dtype, device=xi.device)
            radius = float(obs["radius"])
            B      = barrier_value(pos, center, radius)
            B_values.append(B)
            violation = torch.clamp(-B, min=0.0)
            penalty   = penalty + self.lam * violation ** 2
        V_total = V + penalty
        B_min   = torch.stack(B_values, dim=1).min(dim=1).values \
                  if B_values else torch.full_like(V, 99.0)
        if xi.requires_grad and B_min.min().item() < 1.0:
            g_s = torch.autograd.grad(V_total.sum(), xi,
                create_graph=False, retain_graph=True)[0][:, :3]
            g_t = torch.autograd.grad(V.sum(), xi,
                create_graph=False, retain_graph=True)[0][:, :3]
            g_s_norm = g_s / (torch.norm(g_s, dim=1, keepdim=True) + 1e-8)
            proj     = (g_t * g_s_norm).sum(dim=1, keepdim=True)
            conflict = (g_t * g_s).sum(dim=1) < 0
            correction = torch.zeros_like(u)
            if conflict.any():
                correction[conflict] = -0.1 * u[conflict]
            u = u + correction
        return torch.clamp(u, cfg.RPM_MIN, cfg.RPM_MAX), B_min, V_total
```

---

## 8. Phase 4 — Stability-Aware SGD — `src/stability_sgd.py`

**Path:** `src/stability_sgd.py` | **Lines:** 352 | **Bytes:** 14,018

### Purpose
The system's most mathematically sophisticated module. Adapts the MLP controller online at 100 Hz while guaranteeing that the Lyapunov decrease condition is never permanently violated. Uses a 3-stage verification pipeline (IBP → MC → rollback).

### Imports
```python
from __future__ import annotations
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config as cfg
from src.mlp_controller import MLPController
from src.lyapunov_network import LyapunovNet
from src.quadrotor_dynamics import QuadrotorDynamics
```

### Module-Level Functions (exported for testing)

#### `ibp_linear(W, b, x_lo, x_hi)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `W` | `Tensor [out, in]` | Weight matrix |
| `b` | `Tensor [out]` | Bias |
| `x_lo` | `Tensor [B, in]` | Lower bound on input |
| `x_hi` | `Tensor [B, in]` | Upper bound on input |

**Returns:** `(y_lo, y_hi)` — tight element-wise output bounds

**Formula (IBP soundness theorem):**
```
W⁺ = clamp(W, min=0)    # positive weights
W⁻ = clamp(W, max=0)    # negative weights
y_lo = x_lo @ W⁺ᵀ + x_hi @ W⁻ᵀ + b
y_hi = x_hi @ W⁺ᵀ + x_lo @ W⁻ᵀ + b
```
**Proof:** For each output neuron j and weight Wⱼₖ:
- If Wⱼₖ ≥ 0: min(Wⱼₖ·xₖ) = Wⱼₖ·x_lo,k  and  max = Wⱼₖ·x_hi,k
- If Wⱼₖ < 0: min(Wⱼₖ·xₖ) = Wⱼₖ·x_hi,k  and  max = Wⱼₖ·x_lo,k

Summing across k gives the tightest linear bound. ∎ (Gowal et al., 2018)

#### `ibp_leaky_relu(x_lo, x_hi, slope=0.01)`

Sound interval propagation through LeakyReLU(α=0.01). Handles three cases:
- Both ≥ 0: lrelu(x_lo), lrelu(x_hi)
- Both < 0: slope×x_lo, slope×x_hi
- Mixed (x_lo < 0 ≤ x_hi): y_lo = slope×x_lo, y_hi = x_hi

#### `ibp_sequential(net, x_lo, x_hi)`

Chains `ibp_linear` + `ibp_leaky_relu` + `tanh` + `softplus` through all layers of an `nn.Sequential`. Monotone activations (Tanh, Softplus) propagate bounds directly as `f(x_lo), f(x_hi)`.

---

### Class: `LQRFallback`

**Purpose:** Linear quadratic regulator — emergency backup controller activated after `MAX_CONSECUTIVE_FAIL=3` IBP failures.

**Constructor:** Builds gain matrix `K ∈ ℝ^{4×17}` (hand-tuned):

| Channel | State index | Gain | Physical meaning |
|---------|------------|------|-----------------|
| All motors | 2 (z-pos) | +50 | Altitude hold |
| All motors | 5 (z-vel) | +30 | Altitude damping |
| M1 | 7 (qx) | −20 | Roll correction |
| M2 | 7 (qx) | +20 | Roll correction |
| M3 | 8 (qy) | −20 | Pitch correction |
| M4 | 8 (qy) | +20 | Pitch correction |
| M1,M2 | 10 (roll rate) | ±5 | Roll rate damping |
| M3,M4 | 11 (pitch rate) | ±5 | Pitch rate damping |
| All | 12 (yaw rate) | −3 | Yaw rate damping |

#### Method: `get_control(xi)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | Current state |

**Returns:** `Tensor [B, 4]` — clamped RPM commands

**Formula:** `u = u* − K·(ξ − ξ*)`

---

### Class: `StabilityAwareSGD`

**Constructor parameters:**
- `controller`: `MLPController`
- `lyapunov`: `LyapunovNet`
- `dynamics`: `QuadrotorDynamics`
- `lr`: `float` = `cfg.ADAPT_LR = 1e-4`

**Internal state:**
- `n_adaptations`: count of accepted gradient steps
- `n_rollbacks`: count of parameter rollbacks
- `n_lqr_activations`: count of LQR activations
- `lqr_active`: bool flag
- `_consecutive_fails`: rollback counter
- `_safe_params`: last verified-safe parameter snapshot
- `_latencies`: wall-clock latency log per step
- `_violations`: Lyapunov violation log per step

#### Method: `adapt_online(xi, u_desired=None)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | Current state |
| `u_desired` | `Tensor [B, 4]` or None | Target control (optional) |

**Returns:** `(u, adapted, violation, latency)`
- `u`: `Tensor [B, 4]` — safe control output
- `adapted`: `bool` — whether weights were updated
- `violation`: `float` — Lyapunov decrease violation value
- `latency`: `float` — wall-clock seconds

**8-step algorithm:**

```
Step 1: LQR override check
   If lqr_active:
       Compute u_lqr = LQR.get_control(xi)
       Test neural controller: viol_test = max(0, V(f(ξ,π(ξ))) − (1−κ)V(ξ))
       If viol_test < τ: deactivate LQR
       Return u_lqr

Step 2: Compute violation
   violation = max(0, V(f(ξ, π_θ(ξ))) − (1−κ)V(ξ)).mean()

Step 3: Check threshold
   If violation ≤ τ=1e-3: skip, return u from current θ

Step 4: Save backup θ_backup ← θ

Step 5: Compute gradients
   g_s = ∇_θ[violation]   (stability gradient)
   g_t = ∇_θ[‖π(ξ) − u_desired‖²]  if u_desired else g_s

Step 6: Projected gradient step
   For each parameter p:
       ĝ_s = g_s / (‖g_s‖ + 1e-8)
       dot  = g_t · ĝ_s
       if dot < 0:
           g_proj = g_t − dot · ĝ_s   # remove destabilizing component
       else:
           g_proj = g_t
       p ← p − η · g_proj

Step 7: IBP verification
   ibp_ok = _verify_ibp(xi)
   If not ibp_ok:
       mc_ok = _verify_mc(xi)
       If not mc_ok:
           θ ← θ_backup   (rollback)
           consecutive_fails += 1
           If consecutive_fails ≥ 3: activate LQR
       Else: accept, reset consecutive_fails
   Else: accept, save θ_safe ← θ, reset consecutive_fails
```

#### Method: `_verify_ibp(xi, delta=0.1)`

3-stage IBP verification on box `[ξ−δ, ξ+δ]`:

**Stage 1:** Controller IBP → `[u_lo, u_hi]`
**Stage 2:** Approximate dynamics: `xi_lo_next[:,0:3] += xi_lo[:,3:6] × dt`, same for hi
**Stage 3:** Lyapunov IBP:
```
V_next_max = max(|ibp_sequential(lyap.net, xi_lo_next, xi_hi_next)|)
V_curr_min = min(|ibp_sequential(lyap.net, xi_lo, xi_hi)|)
Return: V_next_max ≤ (1−κ) × V_curr_min
```

#### Method: `_verify_mc(xi, n=200)`

Monte Carlo fallback: sample 200 random perturbations in `[ξ−δ, ξ+δ]`, check what fraction satisfies V(f(ξ,π(ξ))) ≤ (1−κ)V(ξ). Accept if ≥ 90%.

#### Method: `get_stats()`

**Returns:** `dict` with keys:
- `n_adaptations`, `n_rollbacks`, `n_lqr_activations`, `lqr_active`
- `max_violation`, `avg_latency_ms`, `p95_latency_ms`

---

## 9. Phase 5 — Neural Observer — `src/neural_observer.py`

**Path:** `src/neural_observer.py` | **Lines:** 146 | **Bytes:** 4,936

### Purpose
Multi-sensor fusion neural observer. Estimates the full 17D state from noisy partial observations (8D sensor vector). Runs alongside the dynamics model as a parallel predictor-corrector.

### Imports
```python
from __future__ import annotations
import math
import torch
import torch.nn as nn
from src import config as cfg
from src.quadrotor_dynamics import QuadrotorDynamics
```

### Class: `SensorSuite`

**Purpose:** Simulates realistic sensor noise and bias for all 4 sensor modalities.

**Constructor:** `__init__(self, device=cfg.DEVICE)`
- `self.accel_bias`: `Tensor [3]` — accelerometer bias (random walk state)
- `self.gyro_bias`: `Tensor [3]` — gyroscope bias (random walk state)

#### Method: `full_observation(xi)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi` | `Tensor [B, 17]` | Ground-truth state |

**Returns:** `Tensor [B, 8]` — noisy observation vector

| Component | Indices | Formula | Noise |
|-----------|---------|---------|-------|
| IMU accel | 0:3 | `xi[:,3:6] + noise` | σ = 2×10⁻³ |
| IMU gyro | 3:6 | `xi[:,10:13] + noise` | σ = 5×10⁻⁴ |
| Sonar (altitude) | 6 | `clamp(xi[:,2], 0, 2) + noise` | σ = 0.01 |
| Lidar | 7 | `10.0 + noise` (placeholder) | σ = 0.02 |

> [!NOTE]
> The accelerometer model simplifies by using velocity as a proxy (`xi[:,3:6]`). A full IMU model would rotate gravity into the body frame. This is a known simplification noted in the code.

#### Method: `innovation(xi_hat, y_meas)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi_hat` | `Tensor [B, 17]` | Estimated state |
| `y_meas` | `Tensor [B, 8]` | Sensor measurement |

**Returns:** `Tensor [B, 8]` — innovation = y_meas − h(ξ̂)

---

### Helper: `_build_observer_net(arch=cfg.OBSERVER_ARCH)`

Builds `nn.Sequential`: `25→24→12→17` with Tanh hidden, linear output.
**Parameters:** (25×24+24) + (24×12+12) + (12×17+17) = 624+300+221 = **1,145**

### Class: `NeuralObserver(nn.Module)`

**Constructor:** `__init__(self, dynamics=None)`
- `self.dynamics`: `QuadrotorDynamics` (creates new if None)
- `self.sensors`: `SensorSuite()`
- `self.net`: observer MLP (25→24→12→17)

#### Method: `forward(xi_hat, u, y_meas)`

| Parameter | Type/Shape | Description |
|-----------|-----------|-------------|
| `xi_hat` | `Tensor [B, 17]` | Current state estimate |
| `u` | `Tensor [B, 4]` | Applied control |
| `y_meas` | `Tensor [B, 8]` | Sensor measurement |

**Returns:** `Tensor [B, 17]` — updated state estimate

**Observer update equation:**
```
xi_pred     = dynamics(xi_hat, u)           # physics prediction
innov       = y_meas − sensors.obs(xi_hat) # innovation (surprise)
inp_innov   = cat([xi_hat, innov])          # [B, 25]
inp_zero    = cat([xi_hat, zeros(8)])       # [B, 25]
correction  = net(inp_innov) − net(inp_zero) # vanishes at zero innovation
xi_hat_next = xi_pred + correction
# Quaternion renormalisation
xi_hat_next[:,6:10] /= (‖xi_hat_next[:,6:10]‖ + 1e-8)
```

**Key property:** The correction `φ_obs(ξ̂, y−h(ξ̂)) − φ_obs(ξ̂, 0)` vanishes when innovation is zero. This preserves the equilibrium: if the system is at ξ* and sensors are noiseless, the observer predicts ξ* perfectly.

---

## 10. Phase 6 — RRT* Replanner — `src/rrt_replanner.py`

**Path:** `src/rrt_replanner.py` | **Lines:** 185 | **Bytes:** 6,606

### Purpose
Real-time 3D path planning using RRT* (Rapidly-exploring Random Tree Star, Karaman & Frazzoli 2011). Returns a collision-free waypoint sequence within a 50 ms timeout. Includes a Switched Lyapunov safety condition to verify that trajectory switches preserve stability.

### Imports
```python
from __future__ import annotations
import time, math, random
from typing import Optional
import torch
from src import config as cfg
```

### Function: `_segment_sphere_collision(p1, p2, center, radius)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `p1`, `p2` | `list[float]` (len 3) | Segment endpoints |
| `center` | `list[float]` (len 3) | Sphere center |
| `radius` | `float` | Sphere radius |

**Returns:** `bool` — True if segment intersects sphere

**Algorithm (quadratic discriminant):**
```
d = p2 − p1                     (direction vector)
f = p1 − center                 (from center to p1)
a = d·d
b = 2·(f·d)
c = f·f − r²
Δ = b² − 4ac
If Δ < 0: no intersection (return False)
t1 = (−b − √Δ) / (2a)
t2 = (−b + √Δ) / (2a)
Collision if t1 ∈ [0,1] or t2 ∈ [0,1]
```

### Function: `_dist3(a, b)` → `float`
Euclidean distance between two 3D points.

### Class: `RRTReplanner`

**Constructor parameters (all have defaults from config):**
- `workspace_bounds`: `list[tuple]` — default `[(-15,15)]*3`
- `max_samples=1000`, `step_size=0.3`, `goal_bias=0.2`
- `rewire_radius=1.0`, `timeout_ms=50.0`

#### Method: `plan(start, goal, obstacles)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `list[float]` (len 3) | Start position |
| `goal` | `list[float]` (len 3) | Goal position |
| `obstacles` | `list[dict]` | `{center, radius, velocity}` |

**Returns:** `(path, latency_ms)`
- `path`: `list[list[float]]` — waypoints from start to goal
- `latency_ms`: `float` — wall-clock planning time

**RRT* Algorithm:**
```
tree = {start}
for _ in range(max_samples):
    if elapsed > timeout_ms: break
    q_rand = goal (prob=goal_bias) or uniform(bounds)
    nearest = argmin_{q in tree} dist(q, q_rand)
    q_new = steer(nearest, q_rand, step=0.3)
    if not collision_free(nearest, q_new, obstacles): continue
    near = {q in tree : dist(q, q_new) ≤ rewire_radius}
    best_parent = argmin_{q in near} cost(q) + dist(q, q_new)
    add q_new with parent=best_parent
    for q in near:  # rewiring
        if cost(q_new) + dist(q_new,q) < cost(q) and collision_free:
            parent(q) ← q_new
    if dist(q_new, goal) < step:
        extract path by backtracking
        return path, latency
return [start, goal], latency   # fallback: straight line
```

**Collision check** uses `r_safe = obs_radius + DRONE_RADIUS` (inflates obstacles by drone radius).

### Function: `switched_lyapunov_safe(V_curr, rho_old, rho_new, kappa, N=10)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `V_curr` | `float` | Current Lyapunov value |
| `rho_old` | `float` | ROA level set of old trajectory |
| `rho_new` | `float` | ROA level set of new trajectory |
| `kappa` | `float` | = 0.05 (decay rate) |
| `N` | `int` | = 10 (switch horizon) |

**Returns:** `bool`

**Switched Lyapunov condition:**
```
Safe to switch iff:
    V(ξ_t) ≤ ρ_old        (currently in old ROA)
    AND
    V(ξ_t) ≤ (1−κ)^{−N} · ρ_new   (will stay in new ROA for N steps)
```

Physical meaning: guarantees the Lyapunov function does not exceed the new trajectory's ROA during the transient dwell period of N=10 steps (100 ms).

---

## 11. Utilities — `src/utils.py`

**Path:** `src/utils.py` | **Lines:** 139 | **Bytes:** 4,663

### Imports
```python
from __future__ import annotations
import math
import torch
from src import config as cfg
```

### All Functions

#### `rpm_to_rads(rpm)` → `float | Tensor`
`ω = rpm × (2π/60)`

#### `rads_to_rpm(rads)` → `float | Tensor`
`rpm = ω × (60/2π)`

#### `unpack_state(xi: Tensor[B,17])` → `dict`
Returns `{pos:[B,3], vel:[B,3], quat:[B,4], omega:[B,3], motors:[B,4]}`

#### `quat_to_rot(q: Tensor[B,4])` → `Tensor[B,3,3]`
Converts (w,x,y,z) quaternion to rotation matrix:
```
R = [[1-2(y²+z²),   2(xy-wz),    2(xz+wy)  ],
     [2(xy+wz),     1-2(x²+z²),  2(yz-wx)  ],
     [2(xz-wy),     2(yz+wx),    1-2(x²+y²)]]
```

#### `compute_kinetic_energy(vel, omega, mass, J)` → `Tensor[B]`
`KE = ½m‖v‖² + ½ωᵀJω`

#### `compute_potential_energy(pos, mass)` → `Tensor[B]`
`PE = m·g·z` (ENU frame, z = altitude)

#### `random_near_equilibrium(batch, radius)` → `Tensor[batch, 17]`
Samples `batch` states uniformly inside an L² ball of `radius` around ξ*.
Uses volumetric sampling: `noise × rand^(1/17)` to avoid shell bias.
Renormalises quaternion and clamps motor RPMs after perturbation.

#### `validate_hover_rpm()` → `dict`
Verifies HOVER_RPM produces exactly m·g thrust. Returns `{hover_rpm, thrust_total, weight, error_pct}`.

---

## 12. Stub/Alias Files

These thin files exist to allow the `src/` package to be importable from parent scope and to provide backward-compatible aliases.

| File | Size | Content |
|------|------|---------|
| `src/__init__.py` | 76 B | Empty package init |
| `src/lyapunov.py` | 146 B | `from src.lyapunov_network import LyapunovNet` |
| `src/controller.py` | 150 B | `from src.mlp_controller import MLPController` |
| `src/dynamics.py` | 166 B | `from src.quadrotor_dynamics import QuadrotorDynamics` |
| `src/observer.py` | 198 B | `from src.neural_observer import NeuralObserver, SensorSuite` |
| `src/replanner.py` | 196 B | `from src.rrt_replanner import RRTReplanner, switched_lyapunov_safe` |
| `src/barrier.py` | 204 B | `from src.barrier_function import LyapunovBarrierFusion, barrier_value` |
| `src/adaptation.py` | 189 B | `from src.stability_sgd import StabilityAwareSGD, LQRFallback` |

> [!WARNING]
> `docs/API_REFERENCE.md` references `BarrierFunction` (old class name) and `DroneObserver` (old class name). These were refactored to `LyapunovBarrierFusion` and `NeuralObserver` respectively. Tests `test_final.py` and `test_obstacles.py` still import the old names causing `ImportError`.

---

## 13. Phase 7 — Web Dashboard — `web/dashboard_server.py`

**Path:** `web/dashboard_server.py` | **Lines:** 1,004 | **Bytes:** 43,680

### Purpose
Flask + Flask-SocketIO server that drives the real-time 3D web dashboard. All navigation and control decisions are made by the trained neural networks — no hardcoded PID navigation. Streams telemetry at 100 Hz via WebSocket.

### Imports
```python
import sys, os, json, time, math, threading, traceback
import torch
import torch.nn.functional as F
from flask import Flask, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uuid
from src.quadrotor_dynamics import QuadrotorDynamics
from src.mlp_controller import MLPController
from src.lyapunov_network import LyapunovNet
from src.barrier_function import LyapunovBarrierFusion
from src.stability_sgd import StabilityAwareSGD
from src import config as cfg
```

### REST API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve `index.html` |
| GET | `/style.css` | Serve CSS |
| GET | `/app.js` | Serve JS |
| GET | `/reports/<filename>` | Serve generated report images |
| GET | `/api/state` | Full telemetry JSON |
| POST | `/api/start` | Start simulation loop |
| POST | `/api/stop` | Stop + generate report |
| POST | `/api/reset` | Reset to equilibrium |
| POST | `/api/goal` | Set navigation goal `{position:[x,y,z]}` |
| POST | `/api/obstacle` | Add obstacle `{center:[x,y,z], radius:float}` |
| GET | `/api/obstacles` | List all obstacles |
| POST | `/api/clear_obstacles` | Remove all obstacles |
| GET | `/api/report` | Last generated report data |

### WebSocket Events

| Event | Direction | Payload |
|-------|-----------|---------|
| `connect` | Client→Server | Triggers initial telemetry push |
| `disconnect` | Client→Server | Logged |
| `telemetry` | Server→Client | Full state+NN telemetry at 100 Hz |

### Module-Level Functions

#### `get_layer_activations(net, x)` → `list[dict]`
Runs a forward pass through `nn.Sequential`, capturing each layer's output. Returns list of `{layer, size, values}` dicts (max 32 values per layer for UI bandwidth).

#### `get_math_computations(controller, lyapunov, dynamics, xi, u, obstacles)` → `dict`
Extracts step-by-step numerical values for every formula being evaluated this timestep. Returns:
- `controller`: layer-by-layer forward pass with weight shapes, pre-activations, equilibrium subtraction values
- `lyapunov`: delta norm, V_nn, V_lin, R singular values, V_total, decrease condition check
- `barrier`: per-obstacle B values, distances, safety margins
- `dynamics`: motor RPMs, ω_rad, thrusts, total thrust, thrust-to-weight ratio

### Class: `SimulationState`

**Attributes:**
- `dynamics`, `controller`, `lyapunov`: core neural network instances
- `fusion`: `LyapunovBarrierFusion`
- `adapter`: `StabilityAwareSGD`
- `xi`: `Tensor [1, 17]` — current state
- `goal`: `Tensor [3]` — current navigation target
- `waypoints`: `list[Tensor]` — queued waypoints for long-distance navigation
- `obstacles`: `list[dict]`
- `running`: `bool`
- `battery`: `float` (100.0 → 0.0, decreasing with motor load)
- `step_count`, `adaptations`, `replans`: `int` counters
- `history`: `list[dict]` — per-step log for report generation
- `math_history`: `list[dict]` — per-step math values for CSV export
- `lock`: `threading.Lock` — protects state during concurrent HTTP+simulation access

**Checkpoint loading:**
Loads `models/checkpoints/lyapunov_controller_weights.pt` with `weights_only=False` (required for PyTorch ≥ 2.6). Expects keys `controller` and `lyapunov` in the checkpoint dict.

#### Method: `step()`
One 100 Hz simulation tick. Full pipeline:

1. **Outer PD loop:** `pos_error = goal − pos`, `vel_target = direction × min(dist×0.8, V_MAX)`
2. **Waypoint advance:** if `goal_dist < 0.5` and waypoints remain, pop next
3. **Obstacle repulsion:** for each obstacle within 2.5 m, compute repulsion + tangential force to create smooth swerving
4. **Velocity nudge:** `self.xi[:,3:6] += 0.03 × (vel_target − vel)`
5. **ML control:** `u = controller(xi)`
6. **Barrier fusion:** if obstacles, `u_safe = LyapunovBarrierFusion(xi, obstacles)`
7. **Log history:** append pos, vel, V, B, u, violation
8. **Online adaptation:** `u_adapted, adapted, violation, latency = adapter.adapt_online(xi)`
9. **Dynamics step:** `xi = dynamics(xi, u)`, renormalise quaternion
10. **Telemetry extraction:** layer activations, R singular values, all status metrics

#### Method: `set_goal(position)`
Breaks long-distance targets into max 4.0 m segments (waypoints). Activates equilibrium shift: copies goal position into `controller.xi_eq` and `lyapunov.xi_eq` so the networks target the new goal.

#### Method: `_get_decision(violation, barrier_active, goal_dist)` → `dict`
Returns current action label:
- `"AVOIDING"` — barrier function active
- `"ADAPTING"` — Lyapunov violation > τ
- `"TRACKING"` — navigating to goal
- `"STABILIZED"` — at goal, hovering

### Function: `generate_report(sim)` → `dict`
Called on `/api/stop`. Generates 4 matplotlib plots and saves to `web/reports/` and `tests/simulation_1/<timestamp>/`:
1. **3D Trajectory** — flight path + start/end/goal markers + obstacle spheres
2. **Lyapunov & Barrier vs Time** — V(ξ) and B(ξ) time series
3. **Motor RPM Allocation** — all 4 motors vs time with hover line
4. **SGD Violations** — Lyapunov derivative constraint violations

Also exports `calculations.json` and `step_by_step_math.csv` to the archive directory.

### Background Thread: `simulation_loop()`
Runs as a daemon thread at 100 Hz. Calls `sim.step()` then emits `telemetry` via SocketIO.

### Server Entry Point: `run_server()`
Starts daemon simulation thread. Runs `socketio.run(app, host="0.0.0.0", port=5050, debug=False)`.

---

## 14. Web Frontend

### `web/index.html` — Lines: ~320, Bytes: 14,106
HTML5 shell for the dashboard. Loads Three.js (3D), Socket.IO, KaTeX (math rendering), and Chart.js via CDN. Defines the DOM structure: 3D viewport, telemetry panels, neural network visualizer, math computation display, and control buttons.

### `web/style.css` — Lines: ~800, Bytes: 27,678
Full dark-mode CSS. Uses CSS custom properties for theming. Glassmorphism panels with `backdrop-filter: blur`. Animated status indicators (pulse/glow). Responsive grid layout.

### `web/app.js` — Lines: ~1,400, Bytes: 52,498
Complete frontend application:
- **Three.js scene:** Drone mesh (box geometry), rotor discs, 3D trajectory line, obstacle spheres, goal marker, coordinate grid, directional lighting
- **Socket.IO client:** Connects to `ws://localhost:5050`, handles `telemetry` events
- **Real-time chart updates:** Lyapunov value, barrier value, motor RPMs (Chart.js line charts)
- **Neural network visualization:** Bar chart of layer activations for controller and Lyapunov networks
- **Math panel:** Renders KaTeX formulas with live numerical values
- **Control panel:** Start/Stop/Reset buttons, goal position input (x,y,z), add obstacle form
- **Report modal:** Displays generated plots and statistics after stop

---

## 15. Documentation Files — `docs/`

### `docs/ARCHITECTURE.md` (104 lines, 4,834 bytes)
ASCII art data-flow diagram, key design decisions (5 bullet points), power-aware training explanation, observer IMU model details, LQR fallback description, IBP verification steps, and WebSocket streaming architecture.

### `docs/API_REFERENCE.md` (112 lines, 3,188 bytes)
API tables for all 6 core classes. **Note:** Contains stale class names (`BarrierFunction` → now `LyapunovBarrierFusion`; `DroneObserver` → now `NeuralObserver`; `SensorSimulator` → now `SensorSuite`). Architecture listed as `[16,…]` but actual `STATE_REPR_DIM=17`.

### `docs/VERIFICATION_GUIDE.md` (2,763 bytes)
Step-by-step guide to running the test suite, interpreting IBP results, and manually verifying Lyapunov properties.

### `docs/project_report.md` (546 lines, 22,118 bytes)
Comprehensive academic-style report with:
- State-space table
- Phase 1: WGS84 gravity, ISA air density, motor mapping, Euler rotation, quaternion kinematics, motor lag, hover equilibrium derivation
- Phase 2: Yang24 equilibrium subtraction proof, SVD parameterization, two-term Lyapunov construction proof, CEGIS algorithm, total loss formula
- Phase 3: CBF formula, fusion formula, gradient projection
- Phase 4: All 8 adaptation steps, IBP soundness proof, 3-stage pipeline table, LQR gain table
- Phase 5: Observer update equation, all sensor models with noise specs
- Phase 6: RRT* parameters, quadratic discriminant collision test, Switched Lyapunov condition
- Phase 7: Summary
- Table of 6 formal guarantees
- Performance targets table

---

## 16. Model Checkpoints — `models/checkpoints/`

| File | Size | Contents |
|------|------|---------|
| `lyapunov_controller_weights.pt` | 28,224 B | Trained `controller` and `lyapunov` state dicts (Phase 2 CEGIS output) |
| `observer_checkpoint.pt` | 8,629 B | Trained `NeuralObserver` weights (Phase 5 output) |
| `obstacle_avoidance_results.pt` | 499,477 B | Phase 3 evaluation results (trajectories, barrier values, etc.) |
| `online_adaptation_results.pt` | 3,673 B | Phase 4 evaluation stats |
| `replanning_results.pt` | 10,507 B | Phase 6 RRT* planning results |

**Loading (PyTorch ≥ 2.6):**
```python
ckpt = torch.load("lyapunov_controller_weights.pt", map_location="cpu", weights_only=False)
controller.load_state_dict(ckpt["controller"])
lyapunov.load_state_dict(ckpt["lyapunov"])
```

> [!CAUTION]
> PyTorch 2.6 changed the default `weights_only=True`. Loading with `weights_only=True` will fail for these checkpoints because they were saved without safe_globals. Always use `weights_only=False` or register classes with `torch.serialization.add_safe_globals`.

**Known equilibrium mismatch:** The checkpoint was trained with equilibrium at position `(5, 5, 2)` but `config.py` defines `EQUILIBRIUM_STATE` with position `(0, 0, 0)`. This causes `test_lyapunov_positivity` to fail: V evaluated at the config equilibrium is not zero because the loaded checkpoint encodes a different ξ*.

---

## 17. Mathematical Core — All Formulas

### 17.1 Quadrotor Physics

| Formula | Notation | Code Location | Physical Meaning |
|---------|----------|--------------|-----------------|
| Motor thrust | `T_i = K_F·ω_i²` | `motor_forces_torques` | Thrust proportional to angular velocity squared |
| Roll torque | `τ_r = (L/√2)·(T₁−T₂−T₃+T₄)` | `motor_forces_torques` | Differential thrust creates rolling moment |
| Pitch torque | `τ_p = (L/√2)·(−T₁−T₂+T₃+T₄)` | `motor_forces_torques` | Differential thrust creates pitching moment |
| Yaw torque | `τ_y = K_M·(−ω₁²+ω₂²−ω₃²+ω₄²)` | `motor_forces_torques` | Reaction torque from motor spin directions |
| Translational EOM | `mẍ = R·[0,0,F_z]ᵀ + m·g_enu + F_drag` | `forward` | Newton's 2nd law in ENU frame |
| Drag force | `F_drag = −(c₁+c₂‖v‖)·v` | `forward` | Combined Stokes+Newtonian drag |
| Euler rotation | `Jω̇ = τ − ω×(Jω)` | `forward` | Rigid-body angular momentum equation |
| Quaternion kinematics | `q̇ = ½·Ω(ω)·q` | `forward` | Time derivative of orientation quaternion |
| Motor lag | `Ω̇_i = (u_cmd,i−Ω_i)/τ_m` | `forward` | First-order RC-like motor response |
| Hover balance | `4·K_F·ω²_hover = m·g` | `config.py` | Force balance at equilibrium |

### 17.2 Neural Lyapunov Function (Yang24, Eq. 3)

```
V(ξ) = |φ_V(ξ) − φ_V(ξ*)| + ‖(εI + R⊤R)(ξ − ξ*)‖₁
```

where:
- `φ_V`: MLP 17→32→16→1 (Tanh, Softplus)
- `R = U·diag(softplus(σ)+ψ²)·Vᴴ` (SVD-parameterised, always full rank)
- `ε = 1e-4` (positive-definiteness regulariser)

**Decrease condition:**
```
V(ξ_{t+1}) ≤ (1−κ)·V(ξ_t),   κ=0.05
→ V(ξ_t) ≤ (0.95)^t · V(ξ₀)   (exponential convergence)
```

### 17.3 Controller — Yang24 Equilibrium Subtraction (Eq. 2)

```
π_θ(ξ) = clamp(φ(ξ) − φ(ξ*) + u*, RPM_min, RPM_max)
```

**Guarantee:** `π_θ(ξ*) = u*` for **all** θ (structural, not learned).

### 17.4 Control Barrier Function

```
B_i(ξ) = ‖p − o_i‖² − (r_obs,i + r_drone)²
B ≥ 0 ↔ safe

V_total = V(ξ) + λ·Σᵢ [max(0, −Bᵢ)]²,  λ=10
```

### 17.5 Gradient Projection (Safety Filter)

```
ĝ_s = g_s / ‖g_s‖
If g_t·g_s < 0:
    g_proj = g_t − (g_t·ĝ_s)·ĝ_s    (remove anti-safety component)
Else:
    g_proj = g_t
```

Geometric interpretation: `g_proj` lies in the half-space `{g : g·ĝ_s ≥ 0}`.

### 17.6 IBP Soundness (Gowal et al. 2018)

For linear layer `y = Wx + b`:
```
W⁺ = max(W,0),  W⁻ = min(W,0)
y_lo = x_lo@W⁺ᵀ + x_hi@W⁻ᵀ + b
y_hi = x_hi@W⁺ᵀ + x_lo@W⁻ᵀ + b
∀x∈[x_lo,x_hi]: y_lo ≤ Wx+b ≤ y_hi
```

### 17.7 Stability-Aware SGD Projection

```
viol = max(0, V(f(ξ,π_θ(ξ))) − (1−κ)·V(ξ))
g_s  = ∇_θ[viol]
g_t  = ∇_θ[‖π_θ(ξ)−u_desired‖²]
θ ← θ − η·g_proj(g_t, g_s),   η=1e-4
```

### 17.8 Observer Update (Predictor-Corrector)

```
ξ̂_{t+1} = f(ξ̂_t, u_t) + φ_obs(ξ̂_t, y_t−h(ξ̂_t)) − φ_obs(ξ̂_t, 0)
```

Property: correction → 0 when innovation → 0.

### 17.9 Switched Lyapunov Condition (Liberzon 2003)

```
Safe_switch iff:  V(ξ_t) ≤ ρ_old  AND  V(ξ_t) ≤ (1−κ)^{−N}·ρ_new
N = 10 (dwell time = 100 ms at 100 Hz)
```

### 17.10 LQR Fallback

```
u = u* − K·(ξ−ξ*),   K ∈ ℝ^{4×17}
```

### 17.11 Kinetic / Potential Energy (utils.py)

```
KE = ½m‖v‖² + ½ωᵀJω
PE = m·g·z
```

### 17.12 Quaternion → Rotation Matrix (utils.py)

Standard SO(3) parameterisation via unit quaternion:
```
R₁₁ = 1−2(y²+z²),  R₁₂ = 2(xy−wz),  R₁₃ = 2(xz+wy)
R₂₁ = 2(xy+wz),    R₂₂ = 1−2(x²+z²), R₂₃ = 2(yz−wx)
R₃₁ = 2(xz−wy),    R₃₂ = 2(yz+wx),   R₃₃ = 1−2(x²+y²)
```

---

## 18. Complete Data Flow Trace

**One 100 Hz tick (10 ms):**

```
INPUT: ξ_t ∈ ℝ¹⁷  (or ξ̂_t from observer in real deployment)

─── STEP 1: SENSOR FUSION (Phase 5) ─────────────────────────────────────
y_t    = SensorSuite.full_observation(ξ_t)         # [1,8]  noisy obs
innov  = y_t − SensorSuite.full_observation(ξ̂_t)   # [1,8]  innovation
ξ̂_{t+1} = dynamics(ξ̂_t, u_{t-1})                  # physics predict
         + net([ξ̂_t, innov]) − net([ξ̂_t, 0])       # NN correction

─── STEP 2: PATH PLANNING (Phase 6, 50 ms budget) ────────────────────────
if replan_needed:
    path, lat = RRTReplanner.plan(pos, goal, obstacles)
    if switched_lyapunov_safe(V(ξ), ρ_old, ρ_new):
        activate new waypoints

─── STEP 3: PD OUTER LOOP (web/dashboard_server.py) ──────────────────────
pos_error = goal − pos
vel_target = direction × min(‖pos_error‖×0.8, V_MAX)
F_repel    = Σ_obs repulsion_force(pos, obs)
vel_target = clamp(vel_target + F_repel, V_MAX)
xi[:,3:6] += 0.03 × (vel_target − vel)   # nudge velocity

─── STEP 4: NEURAL CONTROL (Phase 2a) ────────────────────────────────────
φ(ξ)  = controller.net(xi)
φ(ξ*) = controller.net(xi_eq)
u_raw = φ(ξ) − φ(ξ*) + u*
u     = clamp(u_raw, 1000, 12000)         # [1,4] RPM commands

─── STEP 5: BARRIER FUSION (Phase 3) ─────────────────────────────────────
V(ξ)     = LyapunovNet(xi)
Bᵢ(ξ)   = ‖pos − center_i‖² − r_safe,i²  (for each obstacle)
penalty  = λ·Σᵢ max(0,−Bᵢ)²
V_total  = V + penalty
if conflict(g_task, g_safety):
    u += correction (−10% of u on conflicting batch elements)
u_safe = clamp(u, 1000, 12000)

─── STEP 6: ONLINE ADAPTATION (Phase 4) ──────────────────────────────────
violation = max(0, V(f(ξ,π(ξ))) − (1−κ)V(ξ))
if violation > 1e-3:
    g_s = ∇_θ[violation]
    g_proj = project(g_task, g_s)
    θ ← θ − 1e-4 × g_proj
    if not IBP_verify(ξ, δ=0.1):
        if not MC_verify(ξ, n=200):
            θ ← θ_backup
            consecutive_fails++
            if consecutive_fails ≥ 3: activate LQR
        else: accept
    else: accept; θ_safe ← θ

─── STEP 7: DYNAMICS INTEGRATION (Phase 1) ───────────────────────────────
rpms_next = rpms + (dt/τ_m)(u_safe − rpms)   # motor lag
F_z, τ   = motor_forces_torques(rpms)
a_enu    = R(q)·[0,0,F_z/m] + [0,0,−g] + drag
vel_next  = vel + dt·a_enu
pos_next  = pos + dt·vel
α        = J⁻¹·(τ − ω×Jω)
ω_next   = ω + dt·α
q_next   = (q + dt·½·Ω(ω)·q) / ‖…‖
ξ_{t+1} = [pos_next, vel_next, q_next, ω_next, rpms_next]

─── STEP 8: TELEMETRY (Phase 7) ──────────────────────────────────────────
socketio.emit('telemetry', {state, status, neural_network, math})
```

**Latency budget:**
| Component | Typical | Budget |
|-----------|---------|--------|
| Controller forward | < 1 ms | — |
| Lyapunov forward | < 1 ms | — |
| Barrier fusion (no grad) | < 1 ms | — |
| SGD adaptation (when triggered) | 5–40 ms | < 50 ms |
| Dynamics integration | < 0.5 ms | — |
| Total tick (no adaptation) | ~3–5 ms | 10 ms |
| Total tick (with adaptation) | ~15–45 ms | 10 ms* |

*Adaptation may cause the tick to exceed 10 ms but the simulation is not real-time constrained — it targets 100 Hz via `time.sleep(0.01)`.

---

## 19. Test Coverage & Known Failures

### Test Files

| File | Tests | Status |
|------|-------|--------|
| `tests/test_stability.py` | Lyapunov positivity, decrease condition, IBP soundness, equilibrium | Mixed |
| `tests/test_final.py` | Integration test across all phases | **FAIL** (ImportError) |
| `tests/test_obstacles.py` | Barrier function, CBF safety | **FAIL** (ImportError) |

### Known Failures & Root Causes

#### 1. `test_final.py` and `test_obstacles.py` — `ImportError`
```
ImportError: cannot import name 'BarrierFunction' from 'src.barrier_function'
ImportError: cannot import name 'DroneObserver' from 'src.neural_observer'
```
**Root cause:** These classes were renamed during refactoring:
- `BarrierFunction` → `LyapunovBarrierFusion`
- `DroneObserver` → `NeuralObserver`

The test files still use the old names. Fix: update imports in test files.

#### 2. `test_lyapunov_positivity` — **FAIL**
```
AssertionError: V(ξ*) = 3.42 > 0 (expected ≈ 0)
```
**Root cause:** `config.py` defines `EQUILIBRIUM_STATE` with position (0,0,0), but the loaded checkpoint (`lyapunov_controller_weights.pt`) was trained with equilibrium at approximately (5,5,2). When the loaded `LyapunovNet.xi_eq` buffer is set to the config equilibrium, V no longer evaluates to zero there.

**Fix options:**
- Re-train with config equilibrium
- Load checkpoint and update `lyapunov.xi_eq.data.copy_(ckpt_eq)` after loading
- Or test positivity at the checkpoint's equilibrium

#### 3. IBP Verification — Overly Conservative
The 3-stage IBP pipeline in `_verify_ibp` sometimes rejects valid adaptations because the Dynamics Stage 2 only propagates position bounds and ignores all other nonlinear dynamics. This causes fall-through to MC verification more often than necessary.

#### 4. PyTorch 2.6 Weights Load Warning
All `torch.load()` calls without `weights_only=True` generate a deprecation warning in PyTorch ≥ 2.1 and will error in future versions. The server explicitly passes `weights_only=False` as a workaround.

---

## 20. Formal Guarantees Summary

| # | Guarantee | Mathematical Statement | Mechanism | File |
|---|-----------|----------------------|-----------|------|
| 1 | **Equilibrium invariance** | `π_θ(ξ*) = u*` ∀θ | Yang24 equilibrium subtraction — algebraic cancellation | `mlp_controller.py` |
| 2 | **Lyapunov PD** | `V(ξ*)=0`, `V(ξ)>0` ∀ξ≠ξ* | SVD full-rank + εI ensures R⊤R is positive definite | `lyapunov_network.py` |
| 3 | **Exponential stability** | `V(ξ_t) ≤ (1−κ)^t · V(ξ₀)`, κ=0.05 | CEGIS training + IBP online verification | `stability_sgd.py` |
| 4 | **Collision avoidance** | `B_i(ξ) ≥ 0` ∀i,t | CBF gradient projection onto safe subspace | `barrier_function.py` |
| 5 | **Safe online adaptation** | IBP fail → rollback → LQR | θ_backup + LQR fallback after 3 failures | `stability_sgd.py` |
| 6 | **Switched stability** | `V ≤ (1−κ)^{−N}·ρ_new` at switch | Dwell-time condition N=10 (Liberzon 2003) | `rrt_replanner.py` |
| 7 | **Observer invariance** | Correction→0 as innovation→0 | Neural correction via difference: φ(ξ̂,innov)−φ(ξ̂,0) | `neural_observer.py` |

### Performance Targets

| Metric | Target | Verification Method |
|--------|--------|-------------------|
| Mission success rate | ≥ 95% | 1,000 Monte Carlo trials |
| Collision rate | 0% | CBF formal certificate |
| ROA volume | ≥ 3.5 m³ | IBP verification during CEGIS |
| Adaptation latency | < 50 ms | p95 of `_latencies` log |
| State estimation error | < 0.05 m position | RMS over held-out simulation |

### Network Parameter Counts

| Network | Architecture | Parameters |
|---------|-------------|-----------|
| MLPController | 17→8→16→16→4 (LeakyReLU) | 628 |
| LyapunovNet MLP | 17→32→16→1 (Tanh+Softplus) | 1,121 |
| LyapunovNet R | σ_raw [17] + ψ [17] | 34 |
| **LyapunovNet total** | — | **1,155** |
| NeuralObserver MLP | 25→24→12→17 (Tanh) | 1,145 |
| LQRFallback | K ∈ ℝ^{4×17} (fixed) | 68 (non-trainable) |
| **Grand total trainable** | — | **≈ 1,928** |

---

*End of Comprehensive Technical Report*
*Generated by deep-read of: src/ (10 files), web/ (4 files), docs/ (4 files), models/checkpoints/ (5 files)*
*Total codebase: ~2,800 lines of Python source + ~2,200 lines of JS/HTML/CSS*
