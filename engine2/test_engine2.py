"""
Standalone test script for Engine 2
====================================
Runs the sensor simulator, tracks detections with the Kalman filter,
predicts 0.5s ahead, and measures error against ground truth.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np

from engine2.sensor_simulator import SensorSimulator
from engine2.kalman_tracker import TrackerManager
from engine2.intent_predictor import IntentPredictor

def main():
    print("=== ENGINE 2: TRACKING & PREDICTION TEST ===")
    
    sim       = SensorSimulator(seed=100)
    tracker   = TrackerManager()
    predictor = IntentPredictor(horizon_s=0.5)
    
    sim.start()
    
    test_duration = 6.0  # seconds
    fps           = 30
    total_frames  = int(test_duration * fps)
    
    print(f"Running simulation for {test_duration} seconds at {fps} Hz...\n")
    
    for frame in range(total_frames):
        # 1. Sense
        detections = sim.sense()
        
        # 2. Track
        tracker.update(detections)
        
        # 3. Predict
        predictions = predictor.predict_all(tracker.active_trackers())
        
        # Print output every 1 second (30 frames)
        if frame % 30 == 0 and frame > 0:
            t = sim.elapsed()
            print(f"--- t = {t:.1f} s ---")
            
            # Ground truth 0.5s in the future
            future_t = t + predictor.horizon_s
            true_future_pos = sim.true_positions(future_t)
            
            for p in predictions:
                true_pos = true_future_pos[p.obs_id]
                error    = IntentPredictor.prediction_error(p, true_pos)
                
                print(f"Obs {p.obs_id}:")
                print(f"  Estimated Vel : {p.velocity.round(3)} m/s")
                print(f"  Predicted Pos : {p.predicted_center.round(3)}")
                print(f"  Actual Future : {true_pos.round(3)}")
                print(f"  Pred Error    : {error:.3f} m")
                
            print()

    print("=== TEST COMPLETE ===")
    
    # Final assertion check for prediction accuracy on types A and B
    print("Final frame prediction errors:")
    t = sim.elapsed()
    future_t = t + predictor.horizon_s
    true_future_pos = sim.true_positions(future_t)
    
    errors = {}
    for p in predictions:
        true_pos = true_future_pos[p.obs_id]
        error    = IntentPredictor.prediction_error(p, true_pos)
        errors[p.obs_id] = error
        print(f"  Type {p.obs_id}: {error:.3f} m")
        
    # We expect linear (A) and curved (B) to be < 0.2m error after 5s of tracking
    assert errors["A"] < 0.2, f"Linear obstacle prediction error {errors['A']:.3f} >= 0.2m"
    assert errors["B"] < 0.4, f"Curved obstacle prediction error {errors['B']:.3f} >= 0.4m"
    print("\nSUCCESS: Prediction error targets met.")

if __name__ == "__main__":
    main()
