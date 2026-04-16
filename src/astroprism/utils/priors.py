"""
priors.py

Utilities for building NIFTy priors from config dicts.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
import nifty8.re as jft

# === Prior registry ===============================================================================

PRIOR_TYPES = {
    "normal":    jft.NormalPrior,
    "lognormal": jft.LogNormalPrior,
    "uniform":   jft.UniformPrior,
    "laplace":   jft.LaplacePrior,
    "invgamma":  jft.InvGammaPrior,
}

# === Main =========================================================================================

def build_prior(name: str, cfg: dict, shape: tuple) -> jft.Model:
    """
    Build a NIFTy prior from a config dict.

    Parameters
    ----------
    name : str
        NIFTy parameter name (used as key in domain).
    cfg : dict
        Prior config dict with keys:
            prior_type   : one of normal, lognormal, uniform, laplace, invgamma
            prior_params : positional args to the prior constructor, either:
                - shared:     [arg0, arg1]               same for all elements
                - per-element [[arg0, arg1], [arg0, arg1], ...]  one per element
    shape : tuple
        Shape of the prior (e.g. (n_channels,) or (n_channels, n_channels)).

    Prior constructor positional args
    ----------------------------------
        normal/lognormal -> [mean, std]
        uniform          -> [a_min, a_max]
        laplace          -> [alpha]
        invgamma         -> [a, scale]
    """
    prior_cls = PRIOR_TYPES[cfg["prior_type"]]
    params = cfg["prior_params"]

    if isinstance(params[0], (list, tuple)):
        # Per-element: [[arg0_el0, arg1_el0], [arg0_el1, arg1_el1], ...]
        # Transpose to get one array per positional arg
        n_args = len(params[0])
        args = [jnp.array([p[i] for p in params]) for i in range(n_args)]
    else:
        # Shared: [arg0, arg1, ...] — broadcast to full shape
        args = [jnp.full(shape, p) for p in params]

    return prior_cls(*args, shape=shape, name=name)
