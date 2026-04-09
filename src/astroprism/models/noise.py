"""
noise.py

Noise model for multi-channel astronomical imaging.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
import nifty8.re as jft
from astroprism.models.priors import build_prior
from astroprism.utils.config import get_defaults

# === Main =========================================================================================

class NoiseModel:
    """
    Heteroscedastic noise model.

    Variance per pixel = background_std^2 + poisson_scale * |signal|
    """

    def __init__(
        self,
        n_channels: int,
        background_std: dict = None,
        poisson_scale: dict = None,
    ):
        p = get_defaults()["params"]
        self.n_channels     = n_channels
        self.background_std = build_prior("background_std", background_std or p["background_std"], (n_channels,))
        self.poisson_scale  = build_prior("poisson_scale",  poisson_scale  or p["poisson_scale"],  (n_channels,))

    @property
    def domain(self) -> dict:
        return self.background_std.domain | self.poisson_scale.domain

    @property
    def init(self) -> dict:
        return self.background_std.init | self.poisson_scale.init

    def __call__(self, x: dict, response: jnp.ndarray) -> jnp.ndarray:
        background_std = self.background_std(x)
        poisson_scale  = self.poisson_scale(x)

        noise_std = []
        for i, channel in enumerate(response):
            bg_var   = background_std[i] ** 2
            ps_var   = poisson_scale[i] * jnp.abs(channel)
            noise_std.append(jnp.sqrt(bg_var + ps_var))

        return noise_std
