"""
Engine 2 — Stage 4: Shared Bus
=====================================
Thread-safe bridge between Engine 2 (writer) and Engine 1 (reader).
Engine 1's `SimulationState` already reads a python list of dicts:
   self.obstacles = [{"center": [x,y,z], "radius": r, "velocity": [vx,vy,vz]}, ...]

This module provides a singleton wrapper around a shared list and lock,
making it easy for Engine 2 to push updates seamlessly.
"""
import threading
from typing import List, Dict

class ObstacleBus:
    """
    Singleton thread-safe shared obstacle dictionary.
    
    Usage in Engine 2:
        bus.update_from_predictions(predictions)
        
    Usage in Engine 1 (Phase 2):
        with bus.lock:
            self.obstacles = bus.get_obstacles_copy()
    """
    _instance = None
    _lock     = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ObstacleBus, cls).__new__(cls)
                cls._instance.lock = threading.Lock()
                # Internal state — mirrors Engine 1's expected format precisely
                cls._instance._obstacles: List[Dict] = []
        return cls._instance
    
    def update_from_predictions(self, predictions: list) -> None:
        """
        Convert a list of ObstaclePrediction objects into the dictionary
        format required by Engine 1 and safely update the bus.
        """
        new_obs_list = []
        for p in predictions:
            new_obs_list.append({
                "center":   p.predicted_center.tolist(),
                "radius":   float(p.radius),
                "velocity": p.velocity.tolist(),
            })
            
        with self.lock:
            self._obstacles = new_obs_list
            
    def get_obstacles_copy(self) -> List[Dict]:
        """
        Return a deep copy of the current obstacle list.
        Engine 1 will call this inside its own tick.
        """
        with self.lock:
            import copy
            return copy.deepcopy(self._obstacles)
