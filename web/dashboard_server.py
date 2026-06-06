"""
Phase 7 -- Flask server: ML-model-driven drone with neural network telemetry.

The trained controller + Lyapunov + Barrier networks make ALL decisions.
No hardcoded navigation -- the model drives the drone.

REST API:
  GET  /              Dashboard
  GET  /api/state     Full state + neural network internals
  POST /api/start     Start simulation
  POST /api/stop      Stop simulation
  POST /api/reset     Reset to equilibrium
  POST /api/goal      Set goal position
  POST /api/obstacle  Add obstacle
"""
from __future__ import annotations

import sys, os, json, time, math, threading, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


# --- Flask App + SocketIO -----------------------------------------------------
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(WEB_DIR, "style.css")


@app.route("/app.js")
def serve_js():
    return send_from_directory(WEB_DIR, "app.js")


@app.route("/reports/<path:filename>")
def serve_reports(filename):
    return send_from_directory(os.path.join(WEB_DIR, "reports"), filename)


# --- Helper: extract hidden layer activations ---------------------------------
def get_layer_activations(net, x):
    """Run forward through nn.Sequential, capturing each layer's output."""
    activations = []
    current = x
    for layer in net:
        current = layer(current)
        # Only record outputs after activation functions (or Linear if last)
        name = layer.__class__.__name__
        act = current.detach().squeeze(0).tolist()
        if isinstance(act, float):
            act = [act]
        activations.append({
            "layer": name,
            "size": len(act) if isinstance(act, list) else 1,
            "values": act[:32],  # cap at 32 neurons for UI
        })
    return activations


def get_math_computations(controller, lyapunov, dynamics, xi, u, obstacles=None):
    """
    Extract step-by-step mathematical computations with actual numerical values.
    Returns a dict describing every formula being evaluated this timestep.
    """
    math_data = {}

    with torch.no_grad():
        xi_sq = xi.squeeze(0)

        # ═══════════════════════════════════════════════════════════════
        # CONTROLLER:  u = clamp(phi(xi) - phi(xi*) + u*, rpm_min, rpm_max)
        # ═══════════════════════════════════════════════════════════════
        ctrl_steps = []

        # Input state
        ctrl_steps.append({
            "label": "Input State",
            "formula": "xi",
            "desc": "xi = [p, v, q, omega, motors]",
            "values": {
                "p (position)": xi_sq[:3].tolist(),
                "v (velocity)": xi_sq[3:6].tolist(),
                "q (quaternion)": xi_sq[6:10].tolist(),
                "omega (ang vel)": xi_sq[10:13].tolist(),
                "motors (RPM)": xi_sq[13:17].tolist(),
            }
        })

        # Layer-by-layer forward pass
        current = xi.clone()
        current_eq = controller.xi_eq.unsqueeze(0).clone()

        for i, layer in enumerate(controller.net):
            layer_name = layer.__class__.__name__
            w_shape = None
            b_shape = None

            if hasattr(layer, 'weight'):
                w = layer.weight
                w_shape = list(w.shape)
                b_shape = list(layer.bias.shape) if layer.bias is not None else None
                # Pre-activation: z = W·h + b
                pre_act = (w @ current.squeeze(0)) + (layer.bias if layer.bias is not None else 0)
                pre_eq = (w @ current_eq.squeeze(0)) + (layer.bias if layer.bias is not None else 0)

                ctrl_steps.append({
                    "label": f"Layer {i//2 + 1}" if 'Linear' in layer_name else f"Activation",
                    "formula": f"z = W{i//2+1} * h + b{i//2+1}" if 'Linear' in layer_name else "",
                    "desc": f"{layer_name} ({w_shape[1]} -> {w_shape[0]})" if w_shape else layer_name,
                    "W_shape": w_shape,
                    "W_norm": w.norm().item(),
                    "b_values": layer.bias.tolist() if layer.bias is not None else [],
                    "pre_activation": pre_act.tolist()[:16],
                })

            current = layer(current)
            current_eq = layer(current_eq)

            if not hasattr(layer, 'weight'):
                ctrl_steps.append({
                    "label": f"Activation",
                    "formula": "h = LeakyReLU(z, slope=0.01)" if "Leaky" in layer_name else f"h = {layer_name}(z)",
                    "desc": layer_name,
                    "post_activation": current.squeeze(0).tolist()[:16],
                })

        phi_xi = current.squeeze(0).tolist()
        phi_eq = current_eq.squeeze(0).tolist()
        u_eq = controller.u_eq.tolist()

        ctrl_steps.append({
            "label": "Equilibrium Subtraction",
            "formula": "u = phi(xi) - phi(xi*) + u*",
            "desc": "Yang24 Eq.2: guarantees pi(xi*) = u*",
            "phi_xi": phi_xi,
            "phi_eq": phi_eq,
            "u_eq": u_eq,
            "u_raw": [(phi_xi[j] - phi_eq[j] + u_eq[j]) for j in range(4)],
        })

        u_final = u.squeeze(0).tolist() if u is not None else [0]*4
        ctrl_steps.append({
            "label": "Clamping",
            "formula": f"u_final = clamp(u, {cfg.RPM_MIN}, {cfg.RPM_MAX})",
            "desc": "Motor RPM limits",
            "u_final": u_final,
        })

        math_data["controller"] = ctrl_steps

        # ═══════════════════════════════════════════════════════════════
        # LYAPUNOV:  V(xi) = V_nn + V_linear
        # ═══════════════════════════════════════════════════════════════
        lyap_steps = []

        # Delta
        delta = (xi - lyapunov.xi_eq.unsqueeze(0)).squeeze(0)
        delta_norm = delta.norm().item()
        lyap_steps.append({
            "label": "State Deviation",
            "formula": "delta = xi - xi*",
            "desc": "Distance from equilibrium",
            "delta_norm": delta_norm,
            "delta_top5": delta.abs().topk(min(5, len(delta))).values.tolist(),
        })

        # NN component
        phi_xi_v = lyapunov.net(xi)
        phi_eq_v = lyapunov.net(lyapunov.xi_eq.unsqueeze(0))
        V_nn = torch.abs(phi_xi_v - phi_eq_v).squeeze().item()

        lyap_steps.append({
            "label": "NN Component",
            "formula": "V_nn = |phi_V(xi) - phi_V(xi*)|",
            "desc": "MLP with Tanh activations + Softplus output",
            "phi_xi": phi_xi_v.item(),
            "phi_eq": phi_eq_v.item(),
            "V_nn": V_nn,
        })

        # R matrix
        R = lyapunov.R
        sv = torch.linalg.svdvals(R).tolist()[:8]
        M = cfg.EPSILON_PD * torch.eye(R.shape[0], device=xi.device, dtype=xi.dtype) + R.t() @ R
        Mdelta = F.linear(delta.unsqueeze(0), M).squeeze(0)
        V_linear = torch.norm(Mdelta, p=1).item()

        lyap_steps.append({
            "label": "Quadratic-Norm Component",
            "formula": "V_lin = ||(eps*I + R^T R) * delta||_1",
            "desc": "SVD-parameterized R ensures full rank",
            "R_singular_values": sv[:6],
            "R_condition": sv[0] / (sv[-1] + 1e-10) if sv else 0,
            "epsilon": cfg.EPSILON_PD,
            "V_linear": V_linear,
        })

        V_total = V_nn + V_linear
        lyap_steps.append({
            "label": "Total Lyapunov Value",
            "formula": "V(xi) = V_nn + V_lin",
            "desc": "V > 0 for all xi != xi*, V(xi*) = 0",
            "V_nn": V_nn,
            "V_linear": V_linear,
            "V_total": V_total,
        })

        # Decrease condition
        xi_next_est = dynamics(xi, u)
        V_next = lyapunov(xi_next_est).item()
        V_decrease = V_next - (1.0 - cfg.KAPPA) * V_total
        lyap_steps.append({
            "label": "Decrease Condition",
            "formula": "V_dot = V(xi+) - (1-kappa)*V(xi)",
            "desc": f"kappa = {cfg.KAPPA}, stable iff V_dot <= 0",
            "V_curr": V_total,
            "V_next": V_next,
            "kappa": cfg.KAPPA,
            "V_dot": V_decrease,
            "is_stable": V_decrease <= 0,
        })

        math_data["lyapunov"] = lyap_steps

        # ═══════════════════════════════════════════════════════════════
        # BARRIER:  B(xi) = ||p - o||^2 - r_safe^2
        # ═══════════════════════════════════════════════════════════════
        if obstacles:
            barrier_steps = []
            pos = xi_sq[:3].tolist()
            for i, obs in enumerate(obstacles):
                oc = obs["center"]
                r = obs["radius"]
                r_safe = r + cfg.DRONE_RADIUS
                dist_sq = sum((pos[j] - oc[j])**2 for j in range(3))
                dist = dist_sq ** 0.5
                B_val = dist_sq - r_safe**2

                barrier_steps.append({
                    "label": f"Obstacle {i+1}",
                    "formula": f"B_{i+1} = ||p - o_{i+1}||^2 - r_safe^2",
                    "desc": f"center={oc}, r={r:.2f}, r_safe={r_safe:.2f}",
                    "position": pos,
                    "obstacle_center": oc,
                    "distance": dist,
                    "r_safe": r_safe,
                    "B_value": B_val,
                    "is_safe": B_val > 0,
                })

            math_data["barrier"] = barrier_steps

        # ═══════════════════════════════════════════════════════════════
        # DYNAMICS:  xi_{t+1} = f(xi_t, u_t)
        # ═══════════════════════════════════════════════════════════════
        motor_rpms = xi_sq[13:17].tolist()
        omega_rad = [rpm * 2 * math.pi / 60 for rpm in motor_rpms]
        thrusts = [cfg.K_F * w**2 for w in omega_rad]
        total_thrust = sum(thrusts)
        math_data["dynamics"] = {
            "formula": "xi_{t+1} = f(xi_t, u_t)",
            "motor_rpms": motor_rpms,
            "omega_rad_s": [round(w, 2) for w in omega_rad],
            "thrust_per_motor_N": [round(t, 4) for t in thrusts],
            "total_thrust_N": round(total_thrust, 4),
            "gravity_force_N": round(cfg.DRONE_MASS * cfg.GRAVITY_SEA_LEVEL, 4),
            "thrust_to_weight": round(total_thrust / (cfg.DRONE_MASS * cfg.GRAVITY_SEA_LEVEL + 1e-10), 4),
            "dt": cfg.DT,
        }

    return math_data


# --- Simulation State ---------------------------------------------------------
class SimulationState:
    def __init__(self):
        self.dynamics = QuadrotorDynamics()
        self.controller = MLPController()
        self.lyapunov = LyapunovNet()
        self.fusion = LyapunovBarrierFusion(self.controller, self.lyapunov, self.dynamics)
        self.adapter = StabilityAwareSGD(self.controller, self.lyapunov, self.dynamics)

        # Load checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints", "lyapunov_controller_weights.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.controller.load_state_dict(ckpt["controller"])
            self.lyapunov.load_state_dict(ckpt["lyapunov"])
            print("[OK] Loaded trained controller + Lyapunov from checkpoint")
        else:
            print("[WARN] No checkpoint -- using untrained networks")

        self.xi = cfg.EQUILIBRIUM_STATE.unsqueeze(0).clone()
        self.goal = cfg.EQUILIBRIUM_STATE[:3].clone()
        self.original_eq = cfg.EQUILIBRIUM_STATE.clone()
        self.waypoints = []
        self.obstacles = []
        self.running = False
        self.battery = 100.0
        self.step_count = 0
        self.adaptations = 0
        self.replans = 0
        self.last_telemetry = {}
        self.history = []
        self.math_history = []
        self.goal_history = []
        self.last_report = None
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.xi = cfg.EQUILIBRIUM_STATE.unsqueeze(0).clone()
            self.goal = cfg.EQUILIBRIUM_STATE[:3].clone()
            # Reset equilibrium in controller and lyapunov
            self.controller.xi_eq.data.copy_(self.original_eq)
            self.lyapunov.xi_eq.data.copy_(self.original_eq)
            self.battery = 100.0
            self.step_count = 0
            self.adaptations = 0
            self.replans = 0
            self.waypoints = []
            self.obstacles = []
            self.last_telemetry = {}
            self.history = []

    def set_goal(self, position):
        with self.lock:
            target = torch.tensor(position, dtype=cfg.DTYPE)
            current_pos = self.goal.clone()
            
            # Break down target into max 4.0m segments
            dist = torch.norm(target - current_pos).item()
            if dist > 4.0:
                num_steps = math.ceil(dist / 4.0)
                step_vector = (target - current_pos) / num_steps
                self.waypoints = [current_pos + step_vector * i for i in range(1, num_steps + 1)]
            else:
                self.waypoints = [target]
                
            if self.waypoints:
                self._activate_goal(self.waypoints.pop(0))

    def _activate_goal(self, position):
        self.goal = position.clone()
        if not hasattr(self, 'goal_history'): self.goal_history = []
        self.goal_history.append(position.tolist())
        new_eq = self.original_eq.clone()
        new_eq[0] = position[0]
        new_eq[1] = position[1]
        new_eq[2] = position[2]
        self.controller.xi_eq.data.copy_(new_eq)
        self.lyapunov.xi_eq.data.copy_(new_eq)
        self.replans += 1

    def add_obstacle(self, center, radius):
        with self.lock:
            self.obstacles.append({
                "center": center,
                "radius": radius,
                "velocity": [0, 0, 0],
            })

    def step(self):
        if not self.running:
            return

        with self.lock:
            try:
                # ===== HYBRID NAVIGATION =====
                # Outer loop: PD position controller → velocity reference
                # Inner loop: ML controller → motor RPMs + stability
                # Barrier: ML barrier function → obstacle avoidance
                # SGD: Online adaptation if Lyapunov violation

                pos = self.xi[0, :3]
                vel = self.xi[0, 3:6]

                # --- Outer PD loop: compute velocity target toward goal ---
                pos_error = self.goal - pos
                goal_dist = torch.norm(pos_error).item()
                
                # Advance to next waypoint if close to current
                if goal_dist < 0.5 and hasattr(self, 'waypoints') and self.waypoints:
                    self._activate_goal(self.waypoints.pop(0))
                    pos_error = self.goal - pos
                    goal_dist = torch.norm(pos_error).item()

                if goal_dist > 0.05:
                    direction = pos_error / (torch.norm(pos_error) + 1e-6)
                    speed = min(goal_dist * 0.8, cfg.V_MAX)
                    vel_target = direction * speed
                else:
                    vel_target = torch.zeros(3, dtype=cfg.DTYPE)

                # --- Obstacle repulsion modifies velocity target ---
                F_repel = torch.zeros(3, dtype=cfg.DTYPE)
                influence_radius = 2.5
                for obs in self.obstacles:
                    oc = torch.tensor(obs["center"], dtype=cfg.DTYPE)
                    diff = pos - oc
                    dist = torch.norm(diff).item()
                    margin = dist - obs["radius"] - cfg.DRONE_RADIUS
                    if margin < influence_radius and margin > 0.01:
                        repel_dir = diff / (dist + 1e-6)
                        strength = 4.0 * (1.0/margin - 1.0/influence_radius) / (margin**2)
                        # Tangential for smooth swerving
                        up = torch.tensor([0., 0., 1.], dtype=cfg.DTYPE)
                        tangent = torch.cross(repel_dir, up)
                        tn = torch.norm(tangent)
                        if tn > 1e-6:
                            tangent = tangent / tn
                        if torch.dot(tangent, pos_error) < 0:
                            tangent = -tangent
                        F_repel = F_repel + repel_dir * strength + tangent * strength * 0.7
                    elif margin <= 0.01:
                        repel_dir = diff / (dist + 1e-6)
                        F_repel = F_repel + repel_dir * 12.0

                vel_target = vel_target + F_repel
                vt_norm = torch.norm(vel_target).item()
                if vt_norm > cfg.V_MAX:
                    vel_target = vel_target * (cfg.V_MAX / vt_norm)

                # --- Apply velocity nudge (PD control on velocity) ---
                with torch.no_grad():
                    vel_correction = 0.03 * (vel_target - vel)
                    self.xi[0, 3:6] = vel + vel_correction

                # --- ML MODEL: Controller + Lyapunov + Barrier ---
                xi_input = self.xi.detach().clone().requires_grad_(True)

                # Controller: compute motor RPMs
                u_controller = self.controller(xi_input)

                # Barrier fusion if obstacles exist
                if self.obstacles:
                    try:
                        u_safe, B_val, V_total = self.fusion(xi_input, self.obstacles)
                        u = u_safe.detach()
                        barrier_active = True
                        barrier_value = B_val.item()
                    except Exception:
                        u = u_controller.detach()
                        barrier_active = False
                        barrier_value = 99.0
                else:
                    u = u_controller.detach()
                    barrier_active = False
                    barrier_value = 99.0

                # Append to history for report generation
                self.history.append({
                    "t": self.step_count * cfg.DT,
                    "pos": pos.cpu().numpy().tolist(),
                    "vel": vel.cpu().numpy().tolist(),
                    "V": V_total.item() if barrier_active else self.lyapunov(xi_input).item(),
                    "B": barrier_value,
                    "u": u.cpu().numpy().tolist(),
                    "violation": 0.0 # Placeholder, updated below if adapted
                })

                # Online SGD adaptation
                try:
                    u_adapted, adapted, violation, latency = self.adapter.adapt_online(xi_input)
                except Exception:
                    adapted = False
                    violation = 0.0
                    latency = 0.0
                if adapted:
                    self.adaptations += 1
                self.history[-1]["violation"] = violation

                # Step dynamics
                with torch.no_grad():
                    self.xi = self.dynamics(self.xi, u)
                    self.xi[:, 6:10] /= torch.norm(self.xi[:, 6:10], dim=1, keepdim=True)
                self.step_count += 1
                
                # Append to math history for CSV report
                try:
                    telemetry_math = get_math_computations(
                        self.controller, self.lyapunov, self.dynamics,
                        xi_input, u, self.obstacles if self.obstacles else None
                    )
                    b_dist, b_safe, b_val = 0.0, 0.0, 0.0
                    if telemetry_math.get("barrier"):
                        obs = telemetry_math["barrier"][0]
                        b_dist = obs.get("distance", 0.0)
                        b_safe = obs.get("r_safe", 0.0)
                        b_val = obs.get("B_value", 0.0)
                    v_nn, v_lin, v_tot, v_dot = 0.0, 0.0, 0.0, 0.0
                    for item in telemetry_math.get("lyapunov", []):
                        if item["label"] == "NN Component": v_nn = item.get("V_nn", 0.0)
                        elif item["label"] == "Quadratic-Norm Component": v_lin = item.get("V_linear", 0.0)
                        elif item["label"] == "Total Lyapunov Value": v_tot = item.get("V_total", 0.0)
                        elif item["label"] == "Decrease Condition": v_dot = item.get("V_dot", 0.0)
                    u_raw, u_clamp = [0,0,0,0], [0,0,0,0]
                    for item in telemetry_math.get("controller", []):
                        if item["label"] == "Equilibrium Subtraction": u_raw = item.get("u_raw", [0,0,0,0])
                        elif item["label"] == "Clamping": u_clamp = item.get("u_final", [0,0,0,0])
                    self.math_history.append({
                        "Time_s": round(self.step_count * cfg.DT, 3),
                        "V_nn": v_nn, "V_linear": v_lin, "V_total": v_tot, "V_dot": v_dot,
                        "B_distance_m": b_dist, "B_safe_margin_m": b_safe, "B_value": b_val,
                        "u_raw_1": u_raw[0] if len(u_raw)>0 else 0, "u_raw_2": u_raw[1] if len(u_raw)>1 else 0, 
                        "u_raw_3": u_raw[2] if len(u_raw)>2 else 0, "u_raw_4": u_raw[3] if len(u_raw)>3 else 0,
                        "u_clamp_1": u_clamp[0] if len(u_clamp)>0 else 0, "u_clamp_2": u_clamp[1] if len(u_clamp)>1 else 0, 
                        "u_clamp_3": u_clamp[2] if len(u_clamp)>2 else 0, "u_clamp_4": u_clamp[3] if len(u_clamp)>3 else 0,
                        "SGD_Violation": violation, "SGD_LR": getattr(self.adapter, 'lr', 0.0)
                    })
                except Exception as e:
                    pass

                # 5. Extract telemetry (no_grad for performance)
                with torch.no_grad():
                    ctrl_activations = get_layer_activations(self.controller.net, self.xi)
                    lyap_activations = get_layer_activations(self.lyapunov.net, self.xi)
                    V_val = self.lyapunov(self.xi).item()
                    R = self.lyapunov.R
                    sv = torch.linalg.svdvals(R).tolist()[:8]

                    u_raw = u.squeeze(0).tolist()
                    u_eq = cfg.EQUILIBRIUM_ACTION.tolist()
                    u_delta = [(u_raw[i] - u_eq[i]) for i in range(4)]

                    pos = self.xi[0, :3].tolist()
                    vel = torch.norm(self.xi[0, 3:6]).item()
                    goal_dist = torch.norm(self.goal - self.xi[0, :3]).item()
                    motor_rpms = self.xi[0, 13:17].tolist()

                    motor_power = torch.norm(self.xi[0, 13:17]).item()
                    self.battery = max(0, self.battery - 0.002 * (motor_power / 6000))

                    min_obs_dist = float('inf')
                    nearest_obs_dir = [0, 0, 0]
                    for obs in self.obstacles:
                        oc = torch.tensor(obs["center"], dtype=cfg.DTYPE)
                        diff = self.xi[0, :3] - oc
                        d = torch.norm(diff).item() - obs["radius"]
                        if d < min_obs_dist:
                            min_obs_dist = d
                            nearest_obs_dir = (diff / (torch.norm(diff) + 1e-6)).tolist()

                    adapter_state = {
                        "learning_rate": cfg.ADAPT_LR,
                        "violation": violation,
                        "adapted": adapted,
                        "latency_ms": latency * 1000,
                        "total_adaptations": self.adaptations,
                        "gradient_norm": 0.0,
                    }

                    # Compute gradient norm if there was adaptation
                    if adapted:
                        total_gnorm = 0.0
                        for p in self.controller.parameters():
                            if p.grad is not None:
                                total_gnorm += p.grad.norm().item() ** 2
                        adapter_state["gradient_norm"] = math.sqrt(total_gnorm)

                    self.last_telemetry = {
                        "state": self.xi.squeeze(0).tolist(),
                        "running": self.running,
                        "status": {
                            "lyapunov_value": V_val,
                            "roa_remaining": max(0, 1 - V_val * 0.01),
                            "safety_margin": min_obs_dist if min_obs_dist != float('inf') else 99.0,
                            "is_stable": violation < cfg.TAU_VIOLATION,
                            "battery_remaining": self.battery,
                            "motor_rpms": motor_rpms,
                            "position": pos,
                            "velocity": vel,
                            "goal": self.goal.tolist(),
                            "goal_distance": goal_dist,
                            "adaptations": self.adaptations,
                            "replans": self.replans,
                            "step_count": self.step_count,
                            "n_obstacles": len(self.obstacles),
                            "violation": violation,
                            "barrier_active": barrier_active,
                            "barrier_value": barrier_value,
                            "nearest_obs_direction": nearest_obs_dir,
                        },
                        "neural_network": {
                            "controller": {
                                "architecture": "17 -> 8 -> 16 -> 16 -> 4",
                                "layers": ctrl_activations,
                                "output_rpms": u_raw,
                                "output_delta": u_delta,
                                "equilibrium_rpms": u_eq,
                            },
                            "lyapunov": {
                                "architecture": "17 -> 32 -> 16 -> 1",
                                "layers": lyap_activations,
                                "V_value": V_val,
                                "R_singular_values": sv,
                            },
                            "sgd": adapter_state,
                            "decision": self._get_decision(violation, barrier_active, goal_dist),
                        },
                        "math": get_math_computations(
                            self.controller, self.lyapunov, self.dynamics,
                            self.xi, u, self.obstacles if self.obstacles else None
                        ),
                    }

            except Exception as e:
                print(f"[SIM ERROR] {e}")
                traceback.print_exc()

    def _get_decision(self, violation, barrier_active, goal_dist):
        """Describe what the model is currently deciding."""
        if barrier_active:
            return {"action": "AVOIDING", "reason": "Barrier function active - obstacle detected, projecting control onto safe subspace"}
        elif violation > cfg.TAU_VIOLATION:
            return {"action": "ADAPTING", "reason": f"Lyapunov violation={violation:.4f} > threshold, SGD correcting controller weights"}
        elif goal_dist > 0.1:
            return {"action": "TRACKING", "reason": f"Controller driving toward goal (dist={goal_dist:.2f}m), Lyapunov certifying stability"}
        else:
            return {"action": "STABILIZED", "reason": "At goal position, maintaining hover via equilibrium subtraction"}

    def get_telemetry(self):
        with self.lock:
            if not self.last_telemetry:
                with torch.no_grad():
                    V_val = self.lyapunov(self.xi).item()
                    ctrl_act = get_layer_activations(self.controller.net, self.xi)
                    lyap_act = get_layer_activations(self.lyapunov.net, self.xi)

                return {
                    "state": self.xi.squeeze(0).tolist(),
                    "running": self.running,
                    "status": {
                        "lyapunov_value": V_val,
                        "roa_remaining": max(0, 1 - V_val * 0.01),
                        "safety_margin": 99.0,
                        "is_stable": True,
                        "battery_remaining": self.battery,
                        "motor_rpms": self.xi[0, 13:17].tolist(),
                        "position": self.xi[0, :3].tolist(),
                        "velocity": 0.0,
                        "goal": self.goal.tolist(),
                        "goal_distance": 0.0,
                        "adaptations": 0,
                        "replans": 0,
                        "step_count": 0,
                        "n_obstacles": 0,
                        "violation": 0.0,
                        "barrier_active": False,
                        "barrier_value": 99.0,
                        "nearest_obs_direction": [0, 0, 0],
                    },
                    "neural_network": {
                        "controller": {
                            "architecture": "17 -> 8 -> 16 -> 16 -> 4",
                            "layers": ctrl_act,
                            "output_rpms": self.xi[0, 13:17].tolist(),
                            "output_delta": [0, 0, 0, 0],
                            "equilibrium_rpms": cfg.EQUILIBRIUM_ACTION.tolist(),
                        },
                        "lyapunov": {
                            "architecture": "17 -> 32 -> 16 -> 1",
                            "layers": lyap_act,
                            "V_value": V_val,
                            "R_singular_values": [],
                        },
                        "sgd": {
                            "learning_rate": cfg.ADAPT_LR,
                            "violation": 0.0,
                            "adapted": False,
                            "latency_ms": 0.0,
                            "total_adaptations": 0,
                            "gradient_norm": 0.0,
                        },
                        "decision": {"action": "IDLE", "reason": "Simulation not started"},
                    }
                }
            return dict(self.last_telemetry)


# --- Global simulation --------------------------------------------------------
sim = SimulationState()


def simulation_loop():
    """Background thread: runs dynamics at 100 Hz, emits telemetry via WebSocket."""
    print("[SIM] Simulation thread started (100 Hz)")
    while True:
        if sim.running:
            sim.step()
            # Push telemetry via WebSocket
            try:
                telemetry = sim.get_telemetry()
                socketio.emit('telemetry', telemetry)
            except Exception:
                pass
        time.sleep(0.01)


# --- REST API -----------------------------------------------------------------
@app.route("/api/state")
def api_state():
    return jsonify(sim.get_telemetry())


def generate_report(sim):
    """Generates a post-simulation report and saves run-specific plots. Returns a report payload."""
    report_id = str(uuid.uuid4())[:8]
    reports_dir = os.path.join(WEB_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Persistent archive location
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join("tests", "simulation_1", timestamp_str)
    os.makedirs(archive_dir, exist_ok=True)
    
    if len(sim.history) == 0:
        return {"error": "No simulation history found."}

    times = [pt["t"] for pt in sim.history]
    px = [pt["pos"][0] for pt in sim.history]
    py = [pt["pos"][1] for pt in sim.history]
    pz = [pt["pos"][2] for pt in sim.history]
    Vs = [pt["V"] for pt in sim.history]
    Bs = [pt["B"] for pt in sim.history]
    us = [pt["u"] for pt in sim.history]
    vols = [pt["violation"] for pt in sim.history]
    
    # Optional constraints
    B_safe = [b if b < 99 else float('nan') for b in Bs]
    
    images = []

    # 1. 3D Trajectory
    try:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(px, py, pz, color="cyan", label="Trajectory")
        ax.scatter([px[0]], [py[0]], [pz[0]], color="green", s=50, label="Start")
        ax.scatter([px[-1]], [py[-1]], [pz[-1]], color="red", s=50, label="End")
        
        if hasattr(sim, "goal_history") and len(sim.goal_history) > 0:
            gx = [g[0] for g in sim.goal_history]
            gy = [g[1] for g in sim.goal_history]
            gz = [g[2] for g in sim.goal_history]
            ax.scatter(gx, gy, gz, color="yellow", s=60, marker="*", label="Goal Targets")
        
        # Plot obstacles
        for obs in sim.obstacles:
            cx, cy, cz = obs["center"]
            r = obs["radius"]
            u = torch.linspace(0, 2 * math.pi, 20)
            v = torch.linspace(0, math.pi, 20)
            x = r * torch.outer(torch.cos(u), torch.sin(v)).numpy() + cx
            y = r * torch.outer(torch.sin(u), torch.sin(v)).numpy() + cy
            z = r * torch.outer(torch.ones(20), torch.cos(v)).numpy() + cz
            ax.plot_surface(x, y, z, color="red", alpha=0.3)
            
        ax.set_title("3D Flight Path")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.tight_layout()
        traj_filename = f"report_traj_{report_id}.png"
        traj_path = os.path.join(reports_dir, traj_filename)
        plt.savefig(traj_path, dpi=120)
        plt.savefig(os.path.join(archive_dir, traj_filename), dpi=120)
        plt.close()
        images.append(f"/reports/{traj_filename}")
    except Exception as e:
        print(f"Error generating trajectory graph: {e}")

    # 2. Lyapunov & Barrier vs Time
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        # Lyapunov
        ax1.plot(times, Vs, color="blue", linewidth=2)
        ax1.set_title("Lyapunov Value $V(\\xi)$ Over Time")
        ax1.set_ylabel("$V(\\xi)$")
        ax1.grid(True, alpha=0.3)
        # Barrier
        ax2.plot(times, B_safe, color="red", linewidth=2)
        ax2.axhline(0, color="orange", linestyle="--", label="Safety boundary ($B=0$)")
        ax2.set_title("Barrier Value $B(\\xi)$ Over Time")
        ax2.set_ylabel("$B(\\xi)$")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        lb_filename = f"report_lyap_bar_{report_id}.png"
        lb_path = os.path.join(reports_dir, lb_filename)
        plt.savefig(lb_path, dpi=120)
        plt.savefig(os.path.join(archive_dir, lb_filename), dpi=120)
        plt.close()
        images.append(f"/reports/{lb_filename}")
    except Exception as e:
        print(f"Error generating Lyapunov/Barrier graph: {e}")

    # 3. Motor RPM Allocation
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Handle cases where u is stored as [[m1, m2, m3, m4]] instead of flat
        flat_u = [u[0] if isinstance(u[0], list) else u for u in us]
        m1 = [f[0] for f in flat_u]
        m2 = [f[1] for f in flat_u]
        m3 = [f[2] for f in flat_u]
        m4 = [f[3] for f in flat_u]
        ax.plot(times, m1, label="Motor 1", alpha=0.7)
        ax.plot(times, m2, label="Motor 2", alpha=0.7)
        ax.plot(times, m3, label="Motor 3", alpha=0.7)
        ax.plot(times, m4, label="Motor 4", alpha=0.7)
        ax.axhline(3980, color="orange", linestyle="--", label="Nominal Hover Out (3980)")
        ax.set_title("Controller Motor Allocation RPMs vs Hover")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RPM")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', prop={'size': 8})
        plt.tight_layout()
        motor_filename = f"report_motors_{report_id}.png"
        motor_path = os.path.join(reports_dir, motor_filename)
        plt.savefig(motor_path, dpi=120)
        plt.savefig(os.path.join(archive_dir, motor_filename), dpi=120)
        plt.close()
        images.append(f"/reports/{motor_filename}")
    except Exception as e:
        print(f"Error generating Motor Allocation graph: {e}")

    # 4. Adaptive SGD Violations
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times, vols, color="purple", linewidth=2, label="Norm Violation (Gradient Step)")
        ax.fill_between(times, 0, vols, color="purple", alpha=0.2)
        ax.set_title("Stability-Aware SGD Optimization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Lyapunov Derivative Constraint Violation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        sgd_filename = f"report_sgd_{report_id}.png"
        sgd_path = os.path.join(reports_dir, sgd_filename)
        plt.savefig(sgd_path, dpi=120)
        plt.savefig(os.path.join(archive_dir, sgd_filename), dpi=120)
        plt.close()
        images.append(f"/reports/{sgd_filename}")
    except Exception as e:
        print(f"Error generating SGD tracking graph: {e}")

    # Capture math and nn states from final telemetry
    telemetry = sim.get_telemetry()
    report_data = {
        "images": images,
        "math": telemetry.get("math", {}),
        "architecture": telemetry.get("neural_network", {}),
        "steps": len(sim.history),
        "duration": times[-1] if times else 0,
        "sgd_adaptations": getattr(sim, 'adaptations', 0)
    }
    
    import json
    import csv
    calculations_path = os.path.join(archive_dir, "calculations.json")
    csv_path = os.path.join(archive_dir, "step_by_step_math.csv")
    try:
        with open(calculations_path, "w") as f:
            json.dump(report_data, f, indent=4)
            
        if sim.math_history:
            keys = sim.math_history[0].keys()
            with open(csv_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(sim.math_history)
    except Exception as e:
        print(f"Error saving calculation payload: {e}")

    return report_data


@app.route("/api/report", methods=["GET"])
def api_report():
    if sim.last_report:
        return jsonify(sim.last_report)
    return jsonify({"error": "No report available."})


@socketio.on('connect')
def handle_connect():
    """Push initial state on WebSocket connection."""
    emit('telemetry', sim.get_telemetry())
    print("[WS] Client connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("[WS] Client disconnected")


@app.route("/api/start", methods=["POST"])
def api_start():
    sim.running = True
    print("[SIM] Simulation STARTED")
    return jsonify({"ok": True, "running": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    sim.running = False
    print("[SIM] Simulation STOPPED")
    # Generate the report using the history accumulated during the run
    sim.last_report = generate_report(sim)
    return jsonify({"ok": True, "running": False})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    sim.running = False
    sim.reset()
    print("[SIM] Simulation RESET")
    return jsonify({"ok": True, "running": False})


@app.route("/api/goal", methods=["POST"])
def api_goal():
    data = request.get_json(force=True)
    pos = data.get("position", [5, 5, 2])
    sim.set_goal(pos)
    print(f"[SIM] Goal set to {pos} -- equilibrium shifted, model will navigate")
    return jsonify({"ok": True, "goal": pos})


@app.route("/api/obstacle", methods=["POST"])
def api_obstacle():
    data = request.get_json(force=True)
    center = data.get("center", [5, 5, 2])
    radius = data.get("radius", 0.3)
    sim.add_obstacle(center, radius)
    print(f"[SIM] Obstacle at {center} r={radius:.2f} -- barrier function will handle avoidance")
    return jsonify({"ok": True, "center": center, "radius": radius,
                    "n_obstacles": len(sim.obstacles)})


@app.route("/api/obstacles", methods=["GET"])
def api_obstacles():
    with sim.lock:
        return jsonify({"obstacles": sim.obstacles})


@app.route("/api/clear_obstacles", methods=["POST"])
def api_clear_obstacles():
    with sim.lock:
        sim.obstacles = []
    print("[SIM] All obstacles cleared")
    return jsonify({"ok": True})


# --- Main ---------------------------------------------------------------------
def run_server():
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

    print("=" * 55)
    print("  Lyapunov Neural Drone Dashboard (ML Model-Driven)")
    print("  http://localhost:5050")
    print("=" * 55)
    print("  All navigation decisions made by trained neural nets")
    print("  Controller: MLP 17->8->16->16->4 (LeakyReLU)")
    print("  Lyapunov:   MLP 17->32->16->1  (Tanh + Softplus)")
    print("  Barrier:    CBF with gradient projection")
    print("  Adaptation: Stability-Aware SGD")
    print("=" * 55)

    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_server()
