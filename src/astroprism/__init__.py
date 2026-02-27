"""AstroPrism: Bayesian inference for multi-channel astronomical imaging."""

import jax

# Enable 64-bit precision for scientific accuracy
jax.config.update("jax_enable_x64", True)

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.dev0"

from astroprism import inference, io, models, operators

__all__ = [
    "__version__",
    "inference",
    "io",
    "models",
    "operators",
]