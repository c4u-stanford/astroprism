"""
reprojection.py

Reprojection operators using pre-computed coordinate grids.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from jax.scipy.ndimage import map_coordinates

# === Main =========================================================================================

class Reprojector:
    """
    Fast image reprojection using pre-computed coordinate grids.

    This class computes the pixel-to-pixel mapping between two WCS systems *once* during initialization.
    The .reproject() method is then a pure JAX interpolation, making it extremely fast and GPU-friendly.
    """

    def __init__(
        self,
        input_wcs: WCS,
        output_wcs: WCS,
        output_shape: tuple[int, int],
        order: int = 1,
        mode: str = "constant",
        cval: float = 0.0,
    ):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.output_shape = output_shape

        # Pre-compute the coordinate grid on CPU (Astropy is CPU-only)
        self.pixel_grid = self._precompute_grid(input_wcs, output_wcs, output_shape)

    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Apply reprojection to an input signal (alias for reproject).
        """
        return self.reproject(s)

    def reproject(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Apply reprojection logic.

        Args:
            s: Input signal array.

        Returns:
            Reprojected signal with shape `output_shape`.
        """
        return map_coordinates(s, self.pixel_grid, order=self.order, mode=self.mode, cval=self.cval)


    def _precompute_grid(self, input_wcs: WCS, output_wcs: WCS, output_shape: tuple[int, int]) -> jnp.ndarray:
        """
        Compute the dense pixel mapping grid (output_pixels -> input_pixels).
        """
        # Extract ny and nx from output_shape
        ny, nx = output_shape

        # Generate grid of output pixel coordinates (y, x)
        # Note: meshgrid 'ij' indexing matches (height, width) order
        y_out, x_out = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

        # Flatten for vectorized astropy call
        x_flat = x_out.ravel()
        y_flat = y_out.ravel()

        # Transform: Output Pixels -> World -> Input Pixels
        # This uses astropy.wcs.utils.pixel_to_pixel which handles distortions/SIP if present
        x_in, y_in = pixel_to_pixel(output_wcs, input_wcs, x_flat, y_flat)

        # Reshape back to grid and stack for map_coordinates
        # map_coordinates expects shape (ndim, *output_shape) -> (2, h, w)
        grid_y = y_in.reshape(ny, nx)
        grid_x = x_in.reshape(ny, nx)

        # Keep as numpy — JAX only captures jax.Array as XLA constants, not numpy arrays.
        # Converting to jnp here would embed this large grid as a constant on every JIT compile.
        return np.stack([grid_y, grid_x])
