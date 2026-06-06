"""Alias module — re-exports StabilityAwareSGD for backward compatibility."""
from src.stability_sgd import StabilityAwareSGD, LQRFallback

__all__ = ["StabilityAwareSGD", "LQRFallback"]
