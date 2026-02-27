"""
response.py

Instrument response model for multi-channel astronomical imaging.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
import nifty8.re as jft
from astropy.wcs import WCS

from astroprism.io import BaseDataset
from astroprism.operators import Convolver, Reprojector

# === Main =========================================================================================

class InstrumentResponse:
    """
    The Forward Model mapping the signal (s) to observed data.

    Model Chain:
    (latent u) -> Signal (s) -> Reprojection (R) -> PSF Convolution (P) -> Data before noise (d_before_noise)

    d_before_noise_i = P_i * R_i(s)

    """

    def __init__(
        self,
        dataset: BaseDataset,
        signal_wcs: WCS,
        signal_shape: tuple[int, int],
        psf_sigma: tuple[float, float] = (0.1, 0.05),
        psf_rotation: tuple[float, float] = (0.0, 1.0),
        reproject_kwargs: dict | None = None,
        convolve_kwargs: dict | None = None,
    ):
        self.dataset = dataset
        self.signal_wcs = signal_wcs
        self.signal_shape = signal_shape
        self.psf_sigma = jft.LogNormalPrior(psf_sigma[0], psf_sigma[1], shape=(dataset.n_channels,), name="psf_sigma")
        self.psf_rotation = jft.NormalPrior(
            psf_rotation[0], psf_rotation[1], shape=(dataset.n_channels,), name="psf_rotation"
        )

        reproject_kwargs = reproject_kwargs or {}
        convolve_kwargs = convolve_kwargs or {}

        # Operators for each channel
        self.reprojectors: list[Reprojector] = []
        self.convolvers: list[Convolver] = []
        for i in range(dataset.n_channels):
            obs_shape = dataset.shapes[i]
            obs_wcs = dataset.wcs[i]
            rp = Reprojector(input_wcs=signal_wcs, output_wcs=obs_wcs, output_shape=obs_shape, **reproject_kwargs)
            cv = Convolver(kernel=dataset.psfs[i], **convolve_kwargs)
            self.reprojectors.append(rp)
            self.convolvers.append(cv)

    @property
    def domain(self) -> dict:
        """Domain of the internal priors"""
        return self.psf_sigma.domain | self.psf_rotation.domain

    @property
    def init(self) -> dict:
        """Init of the internal priors"""
        return self.psf_sigma.init | self.psf_rotation.init

    def __call__(self, x: dict, s: jnp.ndarray) -> list[jnp.ndarray]:
        # TODO: could have more inputs here
        """
        Args:
            x: parameter dict from jft (contains psf_sigma, psf_rotation)
            s: signal field, shape (n_channels, ny_signal, nx_signal)
        Returns:
            list of convolved/reprojected channels (ragged shapes allowed)
        """
        # Get parameters
        psf_sigma = self.psf_sigma(x)
        psf_rotation = self.psf_rotation(x)

        # Apply reprojectors and convolvers
        response = []
        for i in range(self.dataset.n_channels):
            # Get signal
            s_channel = s[i]

            # Reproject
            reprojected_channel = self.reprojectors[i](s_channel)

            # Convolve
            convolved_channel = self.convolvers[i](
                reprojected_channel, kernel_sigma=psf_sigma[i], kernel_rotation=psf_rotation[i]
            )

            # Add to response
            response.append(convolved_channel)

        return response
