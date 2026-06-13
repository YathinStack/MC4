"""
Engine 2 — Main Loop
=====================
Orchestrates the 4 stages of Engine 2 at 30 Hz:
  1. Sensor Simulator (reads raw position)
  2. Kalman Tracker (smooths position, estimates velocity)
  3. Intent Predictor (extrapolates velocity T seconds ahead)
  4. Shared Bus Update (writes predicted state for Engine 1)
"""
import time
import threading

from engine2.sensor_simulator import SensorSimulator
from engine2.kalman_tracker import TrackerManager
from engine2.intent_predictor import IntentPredictor
from engine2.obstacle_bus import ObstacleBus

class Engine2Daemon:
    def __init__(self, horizon_s: float = 0.5, rate_hz: float = 30.0):
        self.rate_hz   = rate_hz
        self.dt        = 1.0 / rate_hz
        
        # Initialise the 4 stages
        self.sensor    = SensorSimulator()
        self.tracker   = TrackerManager(dt=self.dt)
        self.predictor = IntentPredictor(horizon_s=horizon_s)
        self.bus       = ObstacleBus()
        
        self.running   = False
        self._thread   = None
        
    def _loop(self):
        self.sensor.start()
        
        while self.running:
            start_t = time.perf_counter()
            
            # Stage 1: Perception
            detections = self.sensor.sense()
            
            # Stage 2: Tracking (estimate velocity)
            self.tracker.update(detections)
            active_trackers = self.tracker.active_trackers()
            
            # Stage 3: Prediction (linear extrapolate T seconds ahead)
            predictions = self.predictor.predict_all(active_trackers)
            
            # Stage 4: Publish to shared bus
            self.bus.update_from_predictions(predictions)
            
            # Sleep to maintain 30 Hz tick rate
            elapsed = time.perf_counter() - start_t
            sleep_t = self.dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Engine2_Daemon")
        self._thread.start()
        
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
