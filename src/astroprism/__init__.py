import jax

# Enable 64-bit precision for scientific accuracy
jax.config.update("jax_enable_x64", True)

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.dev0"

__all__ = ["__version__"]