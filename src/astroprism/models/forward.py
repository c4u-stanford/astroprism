"""
forward.py

Full forward model for multi-channel astronomical imaging including signal generation, instrument response, and noise model.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
import nifty8.re as jft

# === Main =========================================================================================

class ForwardModel(jft.Model):
    """
    Forward model class for AstroPFM.

    The Full Observation Model combining:
    1. Signal Generation (GP)
    2. Instrument Response (PSF (P) + Reprojection (R))
    3. Noise Model

    d_i = P_i * R_i(s) + n_i

    Returns (mean, std) tuple for Likelihood construction.
    """

    def __init__(
        self,
        signal_model: jft.Model,  # NOTE: could also call signal/source/sky/latent model?
        response_model: jft.Model,
        noise_model: jft.Model,
    ):
        self.signal_model = signal_model
        self.response_model = response_model
        self.noise_model = noise_model

        domain = signal_model.domain | response_model.domain | noise_model.domain
        init = signal_model.init | response_model.init | noise_model.init

        super().__init__(domain=domain, init=init)

    def __call__(self, x: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        # 1. Generate latent signal s from GP parameters
        s = self.signal_model(x)

        # 2. Apply instrument response (needs parameters + signal)
        r = self.response_model(x, s)
        mean_data = r  # TODO: better naming

        # 3. Compute noise standard deviation
        std = self.noise_model(x, r)
        inv_std = [1.0 / std_i for std_i in std] 

        return (mean_data, inv_std)  # TODO: better naming
