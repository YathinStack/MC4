# Verification Guide

Step-by-step instructions for certifying Lyapunov stability.

## 1. Train Controller + Lyapunov

```bash
python training/train_offline.py
```

Watch for:
- **Loss** converging below 0.01
- **Stable fraction** reaching > 95%
- **Region expansion** messages (✓ Expanded region)
- **L_power** decreasing (power-aware penalty converging)

Output: `models/checkpoints/lyapunov_controller_weights.pt`

## 2. Run IBP Verification

```bash
python training/verify_certificates.py
```

Checks Lyapunov decrease at multiple radii δ = {0.1, 0.3, 0.5, trained_radius}.

**PASS criteria**: `stable_fraction ≥ 99.9%` at δ ≤ 0.3

## 3. Run Full Test Suite

```bash
python -m pytest tests/ -v -s
```

### Test Breakdown

| Test | File | Pass Criteria |
|------|------|---------------|
| Quaternion norm | `test_dynamics.py` | \|‖q‖ − 1\| < 10⁻⁶ |
| Motor τ | `test_dynamics.py` | Rise time ∈ [14, 16] ms |
| Energy conservation | `test_dynamics.py` | \|ΔE\|/\|W\| < 0.05 |
| V(ξ*) = 0 | `test_stability.py` | V < 0.01 at equilibrium |
| π(ξ*) = u* | `test_stability.py` | Error < 1 RPM |
| Barrier invariance | `test_obstacles.py` | B(ξ⁺) ≥ 0 for safe states |
| Barrier near obstacle | `test_obstacles.py` | B < 0 inside obstacle |
| Monte Carlo collision | `test_obstacles.py` | < 0.1% collision rate (10k trials) |
| Time to collision | `test_obstacles.py` | 0 < TTC < 10 s |
| Adaptation latency | `test_adaptation.py` | p95 < 50 ms |
| Rollback | `test_adaptation.py` | Params restored exactly |
| IBP soundness | `test_adaptation.py` | True IBP verifies locally |
| Consecutive failures | `test_adaptation.py` | LQR activates after 3 failures |
| Nominal trajectories | `test_final.py` | ≥ 95% success rate |
| Obstacle avoidance | `test_final.py` | 0% collision rate |
| Adaptation speed | `test_final.py` | p95 < 50 ms |
| Observer accuracy | `test_final.py` | Position error < 0.05 m |
| RRT* planning | `test_final.py` | Latency < 100 ms |

## 4. Visual Verification

```bash
python scripts/generate_plots.py
```

Check `results/` for:
- `phase1_quaternion_norm.png` — flat line at 1.0
- `phase2_lyapunov_decrease.png` — all curves decreasing
- `phase3_barrier_values.png` — all values ≥ 0
- `phase4_adaptation.png` — violations recover within 5 steps

## 5. Online Dashboard

```bash
python web/dashboard_server.py
# Open http://localhost:5050
```

Verify:
- **WebSocket**: Status badge shows CONNECTED, telemetry updates in real-time
- LED indicator stays green (stable)
- ROA sphere tracks drone
- Barrier value never goes negative
- Motor bars show balanced thrust at hover
- Power consumption visible in training logs
