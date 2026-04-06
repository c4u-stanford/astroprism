"""
noise.py

Noise model for multi-channel astronomical imaging.
"""

# === Setup ========================================================================================

import jax.numpy as jnp
import nifty8.re as jft
from astroprism.models.priors import build_prior

# === Main =========================================================================================


class NoiseModel:
    """
    Noise model for AstroPFM.

    Model noise as a composition of :
    1. Constant background noise (readout/sky): b^2
    2. Signal Dependent Noise (Poisson-like): gain * |signal|

    Variance = b^2 + gain * |signal|
    """

    def __init__(
        self,
        n_channels: int,
        background_std: dict,
        poisson_scale: dict,
    ):
        self.n_channels = n_channels
        self.background_std = build_prior("background_std", background_std, (n_channels,))
        self.poisson_scale  = build_prior("poisson_scale",  poisson_scale,  (n_channels,))

    @property
    def domain(self) -> dict:
        """Domain of the internal priors"""
        return self.background_std.domain | self.poisson_scale.domain

    @property
    def init(self) -> dict:
        """Init of the internal priors"""
        return self.background_std.init | self.poisson_scale.init

    def __call__(self, x: dict, response: jnp.ndarray) -> jnp.ndarray:
        # Get parameters
        background_std = self.background_std(x)
        poisson_scale = self.poisson_scale(x)

        # Variance components
        noise_std = []
        for i, channel in enumerate(response):
            bg_var = background_std[i]**2
            ps_var = poisson_scale[i] * jnp.abs(channel)
            noise_var = bg_var + ps_var
            noise_std.append(jnp.sqrt(noise_var))

        return noise_std


#  """Domain of the internal priors"""
#         return self.psf_sigma.domain | self.psf_rotation.domain


# class NoiseModel(jft.Model):
#     """
#     Learnable heteroscedastic noise model for pre-processed astronomical data.

#     For each band, learns two parameters that capture the main noise contributions:
#     - background: Background noise standard deviation (flux-independent noise)
#     - signal_gain: Signal-dependent noise scaling (flux-dependent noise)

#     Total noise variance per pixel = background^2 + signal_gain * |signal|

#     This parameterization learns the effective noise characteristics rather than
#     modeling explicit detector physics, making it suitable for pre-processed data.
#     """

#     def __init__(self, n_channels,
#                  background_prior=(0.01, 0.1),      # (mean, std) for background noise std
#                  signal_gain_prior=(0.01, 0.1)):    # (mean, std) for signal-dependent gain

#         self.n_channels = n_channels

#         # Learn background noise std per band (flux-independent noise)
#         self.background = jft.LogNormalPrior(
#             background_prior[0], background_prior[1],
#             shape=(n_channels,), name="background"
#         )

#         # Learn signal-dependent gain per band (noise scaling with signal)

#         self.signal_gain = jft.LogNormalPrior(
#             signal_gain_prior[0], signal_gain_prior[1],
#             shape=(n_channels,), name="signal_gain"
#         )

#         domain = self.background.domain | self.signal_gain.domain
#         init = self.background.init | self.signal_gain.init
#         super().__init__(domain=domain, init=init)

#     def __call__(self, x):
#         """
#         Return the noise parameters for the given input.
#         This is required by NIFTy8's Model interface.
#         """
#         return {
#             'background': self.background(x),
#             'signal_gain': self.signal_gain(x)
#         }

#     def noise_variance(self, x, signal_response):
#         """
#         Compute noise variance for each pixel.

#         Args:
#             x: noise parameters (background, signal_gain)
#             signal_response: model signal response prediction (n_channels, ny, nx)

#         Returns:
#             variance: noise variance for each pixel (n_channels, ny, nx)
#         """
#         # Get learned parameters
#         background = self.background(x) # (n_channels,) or scalar
#         signal_gain = self.signal_gain(x) # (n_channels,) or scalar

#         # Ensure parameters are arrays with correct shape
#         if background.ndim == 0:  # scalar
#             background = jnp.full((self.n_channels,), background)
#         if signal_gain.ndim == 0:  # scalar
#             signal_gain = jnp.full((self.n_channels,), signal_gain)

#         # Compute noise variance components
#         background_var = background[:, None, None]**2  # (n_channels, 1, 1) - broadcast to all pixels
#         # Heteroscedastic component scales linearly with signal magnitude
#         signal_var = signal_gain[:, None, None] * jnp.abs(signal_response) # (n_channels, ny, nx)

#         # Total variance = background^2 + signal_gain * |signal|
#         total_variance = background_var + signal_var

#         return total_variance
