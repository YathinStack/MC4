"""Alias module — re-exports barrier utilities for backward compatibility."""
from src.barrier_function import LyapunovBarrierFusion, barrier_value

__all__ = ["LyapunovBarrierFusion", "barrier_value"]
