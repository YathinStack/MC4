# API Reference

All tensors are PyTorch `torch.Tensor`. Batch dimension `B` is always first.

---

## `src/quadrotor_dynamics.py`

### `QuadrotorDynamics(dt=0.01)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ, u_cmd)` | ξ:[B,17], u:[B,4] | ξ_next:[B,17] | RPM |
| `motor_forces_torques(rpm)` | rpm:[B,4] | F_z:[B], τ:[B,3] | N, N·m |

---

## `src/mlp_controller.py`

### `MLPController(arch=[17, 8, 16, 16, 4])`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ)` | ξ:[B,17] | u:[B,4] | RPM ∈ [1000, 12000] |

---

## `src/lyapunov_network.py`

### `LyapunovNet(arch=[17, 32, 16, 1])`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ)` | ξ:[B,17] | V:[B] | dimensionless, ≥ 0 |

---

## `src/barrier_function.py`

### `barrier_value(pos, obs_center, obs_radius)` (module-level function)

| Parameter | Type | Description |
|-----------|------|-------------|
| `pos` | Tensor [B,3] | Drone position |
| `obs_center` | Tensor [3] | Obstacle center |
| `obs_radius` | float | Obstacle radius |
| **Returns** | Tensor [B] | B(ξ) = ‖p−o‖² − r_safe²; ≥ 0 = safe |

### `LyapunovBarrierFusion(controller, lyapunov, dynamics)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ, obstacles)` | ξ:[B,17], list[dict] | u:[B,4], B:[B], V:[B] | — |

---

## `src/stability_sgd.py`

### `StabilityAwareSGD(controller, lyapunov, dynamics, lr=1e-4)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `adapt_online(ξ, u_desired)` | ξ:[1,17], u:[1,4] | u, adapted, violation, latency | RPM, bool, float, s |
| `get_stats()` | — | dict | — |

---

## `src/neural_observer.py`

### `NeuralObserver(dynamics)` (was `DroneObserver`)

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ̂, u, y_meas)` | [B,17], [B,4], [B,8] | ξ̂_next:[B,17] | — |

### `SensorSuite(device)` (was `SensorSimulator`)

| Method | Returns | Rate |
|--------|---------|------|
| `full_observation(ξ)` | [B,8] (accel+gyro+sonar+lidar) | 100 Hz |
| `innovation(ξ_hat, y_meas)` | [B,8] | — |

---

## `src/rrt_replanner.py`

### `RRTReplanner(workspace_bounds, max_samples, step_size, goal_bias, rewire_radius, timeout_ms)` (was `Replanner`)

| Method | Args | Returns |
|--------|------|---------|
| `plan(start, goal, obstacles)` | list[float] [3], list[float] [3], list[dict] | (path: list[list[float]], latency_ms: float) |

### `switched_lyapunov_safe(V_curr, rho_old, rho_new, kappa, N)` (module-level function)

| Args | Returns |
|------|---------|
| floats | bool — True if safe to switch trajectory |

---

## Obstacle Dict Format

```python
{
    "center": [x, y, z],    # meters (ENU)
    "radius": float,        # meters
    "velocity": [vx, vy, vz]  # m/s (optional, default [0,0,0])
}
```
