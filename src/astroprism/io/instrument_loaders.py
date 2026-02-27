"""
instrument_loaders.py

Instrument-specific FITS loaders for different telescopes.
"""

# === Imports ======================================================================================

import numpy as np
import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS

# === Main =========================================================================================

# TODO: also need things like position angle, units etc. - maybe also have an hdf5 loader

def _load_jwst(path: str) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
    """Load data, WCS, and PSF from a JWST FITS file.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        psf = hdul["PSF"].data.astype(np.float64)
    return data, wcs, psf


def _load_hst(path: str) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
    """Load data, WCS, and PSF from a HST FITS file.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        psf = hdul["PSF"].data.astype(np.float64)
    return data, wcs, psf


def _load_astrosat(path: str) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
    """Load data, WCS, and PSF from an Astrosat FITS file.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        psf = hdul["PSF"].data.astype(np.float64)
    return data, wcs, psf


def _load_spitzer(path: str) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
    """Load data, WCS, and PSF from a Spitzer FITS file.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        psf = hdul["PSF"].data.astype(np.float64)
    return data, wcs, psf


def _load_herschel(path: str) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
    """Load data, WCS, and PSF from a Herschel FITS file.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        wcs = WCS(hdul[0].header)
        psf = hdul["PSF"].data.astype(np.float64)
    return data, wcs, psf


_INSTRUMENT_LOADERS = {
    "JWST_MIRI": _load_jwst,
    "JWST_NIRCAM": _load_jwst,
    "HST_WFC3": _load_hst,
    "ASTROSAT": _load_astrosat,
    "SPITZER_IRAC": _load_spitzer,
    "HERSCHEL_PACS": _load_herschel,
    "HERSCHEL_SPIRE": _load_herschel,
}