# Online Adaptive Lyapunov-Stable Drone Control System

Production-ready quadrotor control with formally verified Lyapunov stability, CBF obstacle avoidance, online adaptation, sensor fusion, and a real-time 3D web dashboard.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all phases (training + tests + plots)
python scripts/run_pipeline.py

# Or run phases individually:
python training/train_lyapunov_controller.py      # Phase 2: CEGIS training
python training/train_observer.py                 # Phase 5: Observer training
python scripts/generate_plots.py                  # Generate all visualisations
python -m pytest tests/ -v -s         # Run all tests

# Launch web dashboard
python web/dashboard_server.py
# Open http://localhost:5000
```

## Architecture

| Module | Phase | Description |
|--------|-------|-------------|
| `src/quadrotor_dynamics.py` | 1 | 17D quadrotor dynamics (WGS84 gravity, ISA density, X-config motors) |
| `src/mlp_controller.py` | 2 | MLP controller with Yang24 equilibrium subtraction |
| `src/lyapunov_network.py` | 2 | Neural Lyapunov function with 1-norm + SVD positivity |
| `src/barrier_function.py` | 3 | CBF obstacle avoidance with gradient projection |
| `src/stability_sgd.py` | 4 | Stability-aware SGD + IBP local verification |
| `src/neural_observer.py` | 5 | NN sensor fusion (IMU/GPS/Lidar/Sonar) |
| `src/rrt_replanner.py` | 6 | RRT* path planning with switched Lyapunov |
| `web/` | 7 | Three.js dashboard + Flask/WS backend |

## State Space

```
ξ ∈ ℝ¹⁷ = [p₃, v₃, q₄, ω₃, Ω₄]
         = [position, velocity, quaternion, angular_rate, motor_rpms]
```

## Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Mission success | ≥ 95% | Monte Carlo (1000 trials) |
| Collision rate | 0% | Formally certified via CBF |
| ROA volume | ≥ 3.5 m³ | IBP verification |
| Adaptation latency | < 50 ms | Stability-aware SGD |
| Estimation error | < 0.05 m | NN observer |
