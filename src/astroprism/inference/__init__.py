"""Inference algorithms for posterior estimation."""

from astroprism.inference.vi import VariationalInference, run_inference

__all__ = [
    "VariationalInference",
    "run_inference",
]
