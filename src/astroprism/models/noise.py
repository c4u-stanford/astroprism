"""
noise.py

Noise model for multi-channel astronomical imaging.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
import nifty8.re as jft
from astroprism.utils.priors import build_prior
from astroprism.utils.config import get_defaults

# === Main =========================================================================================

class NoiseModel:
    """
    Heteroscedastic noise model.

    Variance per pixel = noise_floor^2 + noise_scale * |signal|
    """

    def __init__(
        self,
        n_channels: int,
        noise_floor: dict = None,
        noise_scale: dict = None,
    ):
        p = get_defaults()["noise"]
        self.n_channels     = n_channels
        self.noise_floor = build_prior("noise_floor", noise_floor or p["noise_floor"], (n_channels,))
        self.noise_scale  = build_prior("noise_scale",  noise_scale  or p["noise_scale"],  (n_channels,))

    @property
    def domain(self) -> dict:
        return self.noise_floor.domain | self.noise_scale.domain

    @property
    def init(self) -> dict:
        return self.noise_floor.init | self.noise_scale.init

    def __call__(self, x: dict, response: jnp.ndarray) -> jnp.ndarray:
        noise_floor = self.noise_floor(x)
        noise_scale  = self.noise_scale(x)

        noise_std = []
        for i, channel in enumerate(response):
            bg_var   = noise_floor[i] ** 2
            ps_var   = noise_scale[i] * jnp.abs(channel)
            noise_std.append(jnp.sqrt(bg_var + ps_var))

        return noise_std
