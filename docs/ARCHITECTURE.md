# Architecture

## System Data Flow

```
Sensors (IMU/GPS/Lidar/Sonar)
        │
        ▼
  ┌─────────────┐
  │  Observer    │  NN state estimator (MLP 25→24→12→17)
  │  (Phase 5)  │  ξ̂_{t+1} = f(ξ̂_t, u_t) + φ(ξ̂, innovation)
  └──────┬──────┘
         │ ξ̂_t (estimated state)
         ▼
  ┌─────────────┐    obstacles    ┌──────────────┐
  │  Replanner  │◄───────────────│  Lidar/Map   │
  │  (Phase 6)  │  RRT*          └──────────────┘
  └──────┬──────┘
         │ ξ_ref(t) (reference trajectory)
         ▼
  ┌─────────────┐
  │  Controller │  MLP (17→8→16→16→4)
  │  (Phase 2)  │  π_θ(ξ) = φ(ξ) − φ(ξ*) + u*
  └──────┬──────┘
         │ u_cmd (motor RPMs)
         ▼
  ┌─────────────┐    violation?   ┌──────────────┐
  │  Adaptation │◄───────────────│  Lyapunov    │
  │  (Phase 4)  │  projected SGD │  (Phase 2)   │
  └──────┬──────┘                │  V(ξ) ≥ 0   │
         │ u_safe                └──────┬───────┘
         ▼                              │
  ┌─────────────┐                ┌──────▼───────┐
  │  Barrier    │◄──────────────│  IBP Verifier│
  │  (Phase 3)  │  B(ξ) ≥ 0    └──────────────┘
  └──────┬──────┘
         │ u_final (safety-filtered)
         ▼
  ┌─────────────┐
  │  Dynamics   │  17D quadrotor physics (STATE_REPR_DIM=17)
  │  (Phase 1)  │  ξ_{t+1} = f(ξ_t, u_t)
  └─────────────┘
```

## Key Design Decisions

1. **Equilibrium Subtraction** (Yang24): Controller output equals hover RPMs at equilibrium by construction
2. **SVD Parameterisation**: Lyapunov R matrix is always full-rank → V(ξ) > 0 guaranteed
3. **Gradient Projection**: When safety and performance conflict, project task gradient onto safe subspace
4. **IBP Rollback**: Online adaptation reverts if local verification fails → never loses stability
5. **Switched Lyapunov**: Replanning preserves stability via dwell-time condition at trajectory switches

## Power-Aware Training (P2)

The CEGIS training loop includes a power-consumption penalty:

```
L_power = C3_POWER_WEIGHT · mean(P(ξ, u) / P_hover)
```

- `P(ξ, u)` is the electrical power computed from motor RPMs and their rates via `dynamics.motor_power()`
- Normalised by hover power `P_hover` for a dimensionless loss
- Encourages the controller to find energy-efficient trajectories during CEGIS training

## Observer IMU Model

The `SensorSimulator.imu()` computes physically correct accelerometer readings:

```
a_meas = (F_thrust / m) · ê_z_body − Rᵀ·g + b_a + n_a
```

- Thrust is derived from current motor RPMs in the state vector: `F_z = Σ k_f · ω_i²`
- Specific force is thrust/mass along body z-axis minus gravity rotated into body frame
- Includes bias random walks and additive Gaussian noise per the sensor noise parameters

## LQR Fallback Controller (P1)

When online adaptation fails 3+ consecutive times:

1. Controller parameters are reverted to the last verified-safe checkpoint
2. An LQR fallback activates: `u = u_eq − K · (ξ − ξ_eq)`
3. The diagonal gain matrix `K` is tuned for hover stabilisation
4. The neural controller is periodically re-checked; LQR deactivates once it recovers

## True IBP Verification (P1)

Online adaptation uses sound Interval Bound Propagation:

1. **Controller IBP**: Propagate `[ξ−δ, ξ+δ]` through the MLP using W+/W− decomposition
2. **Dynamics intervals**: Bound the next-state region using interval arithmetic
3. **Lyapunov IBP**: Verify `V(ξ_next_max) − (1−κ)·V(ξ_curr_min) ≤ 0`

Falls back to sampling-based verification (200 Monte Carlo samples) when IBP is too conservative.

## WebSocket Streaming (P3)

The web dashboard uses **Flask-SocketIO** for real-time telemetry:

- Server emits `telemetry` events at 100 Hz via WebSocket
- Client connects via Socket.IO with automatic transport negotiation
- Falls back to HTTP polling (`/api/state`) if WebSocket is unavailable
- REST API (`/api/start`, `/api/stop`, `/api/reset`, `/api/goal`, `/api/obstacle`) remains for control actions
