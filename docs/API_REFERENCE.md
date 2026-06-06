# API Reference

All tensors are PyTorch `torch.Tensor`. Batch dimension `B` is always first.

---

## `src/quadrotor_dynamics.py`

### `QuadrotorDynamics(dt=0.01)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ, u_cmd)` | ξ:[B,16], u:[B,4] | ξ_next:[B,16] | RPM |
| `gravity(alt_m)` | alt_m:Tensor | g:Tensor | m/s² |
| `air_density(alt_m)` | alt_m:Tensor | ρ:Tensor | kg/m³ |
| `motor_forces_torques(rpm)` | rpm:[B,4] | F_z:[B,1], τ:[B,3] | N, N·m |
| `motor_power(rpm, rpm_dot)` | rpm:[B,4], rpm_dot:[B,4] | P:[B] | W |
| `simulate(ξ0, u_fn, steps)` | ξ0:[B,16], callable, int | traj:[T+1,B,16] | — |

---

## `src/mlp_controller.py`

### `MLPController(arch=[16,8,16,16,4])`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ)` | ξ:[B,16] | u:[B,4] | RPM ∈ [1000, 12000] |

---

## `src/lyapunov_network.py`

### `LyapunovNet(arch=[16,32,16,1])`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ)` | ξ:[B,16] | V:[B] | dimensionless, ≥ 0 |
| `decrease_condition(V_curr, V_next)` | V:[B], V:[B] | violation:[B] | ≥ 0 means violated |

---

## `src/barrier_function.py`

### `BarrierFunction(drone_radius=0.2)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ, obstacles)` | ξ:[B,16], list[dict] | B_min:[B] | m², ≥0 = safe |
| `time_to_collision(ξ, obs)` | ξ:[B,16], dict | TTC:[B] | seconds |

### `LyapunovBarrierFusion(controller, lyapunov, dynamics)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ, obstacles)` | ξ:[B,16], list[dict] | u:[B,4], B:[B], V:[B] | — |

---

## `src/stability_sgd.py`

### `StabilityAwareSGD(controller, lyapunov, dynamics)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `adapt_online(ξ, u_desired)` | ξ:[1,16], u:[1,4] | u, adapted, violation, latency | RPM, bool, float, s |
| `get_stats()` | — | dict | — |

---

## `src/neural_observer.py`

### `DroneObserver(dynamics)`

| Method | Args | Returns | Units |
|--------|------|---------|-------|
| `forward(ξ̂, u, y_meas)` | [B,16], [B,4], [B,8] | ξ̂_next:[B,16] | — |

### `SensorSimulator(device)`

| Method | Returns | Rate |
|--------|---------|------|
| `imu(ξ_true)` | [B,6] (accel+gyro) | 100 Hz |
| `gps(ξ_true)` | [B,3] (position) | 10 Hz |
| `lidar_min_range(ξ_true, obs)` | [B,1] (range) | 30 Hz |
| `sonar(ξ_true)` | [B,1] (altitude) | 50 Hz |
| `full_observation(ξ_true, obs)` | [B,8] | — |

---

## `src/rrt_replanner.py`

### `Replanner(dynamics, controller, lyapunov)`

| Method | Args | Returns |
|--------|------|---------|
| `replan(ξ, goal, obstacles)` | [1,16], [3], list | list[[3]] waypoints |
| `switch_controller_safely(ξ, ρ_old, ρ_new)` | [1,16], float, float | bool |
| `get_stats()` | — | dict |

---

## Obstacle Dict Format

```python
{
    "center": [x, y, z],    # meters (ENU)
    "radius": float,        # meters
    "velocity": [vx, vy, vz]  # m/s (optional, default [0,0,0])
}
```
