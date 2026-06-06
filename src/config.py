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
