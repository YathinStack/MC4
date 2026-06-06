"""Alias module — re-exports RRTReplanner for backward compatibility."""
from src.rrt_replanner import RRTReplanner, switched_lyapunov_safe

__all__ = ["RRTReplanner", "switched_lyapunov_safe"]
