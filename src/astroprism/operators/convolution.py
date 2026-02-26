"""
convolution.py

PSF convolution operator with learnable kernel parameters.
"""

# === Imports ======================================================================================

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax.scipy.signal import convolve, fftconvolve

# === Main =========================================================================================

class Convolver:
    """
    Convolver class. This class is used to convolve an image with a PSF.
    """

    def __init__(self, kernel: jnp.ndarray, mode: str = "same", axes: tuple | None = None):
        self.kernel = kernel
        self.mode = mode
        self.axes = axes

    # @partial(jax.jit, static_argnums=(1, 2))
    def __call__(self, s: jnp.ndarray, kernel_sigma: float = 1.0, kernel_rotation: float = 0.0) -> jnp.ndarray:
        # Rotate kernel
        kernel = rotate_kernel(self.kernel, kernel_rotation)

        # Convolve kernel with Gaussian
        kernel = gaussian_filter_jax(kernel, sigma=kernel_sigma)

        # Apply circular mask
        kernel = apply_circular_mask(kernel)

        # Normalize kernel
        kernel = kernel / jnp.maximum(kernel.sum(), 1e-12)

        return self.convolve(s, kernel)

    def convolve(self, s: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        return fftconvolve(s, kernel, mode=self.mode, axes=self.axes)


# === Utilities ====================================================================================


def gaussian_filter1d_jax(array: jnp.ndarray, sigma: float, axis: int = 0, truncate: float = 4.0) -> jnp.ndarray:
    """1D Gaussian filter along specified axis."""

    # TODO: this is hardcoded, what do i do here?
    # Fixed maximum to support sigma up to ~12
    MAX_RADIUS = 50
    MAX_SIZE = 2 * MAX_RADIUS + 1

    # Clip sigma for numerical stability
    sigma = jnp.clip(sigma, 1e-6, None)  # TODO: maybe change to 1e-12

    # Compute actual radius (but capped)
    radius = jnp.minimum(truncate * sigma, MAX_RADIUS)

    # Create full kernel
    x = jnp.arange(MAX_SIZE, dtype=float) - MAX_RADIUS
    kernel = jnp.exp(-(x**2) / (2 * sigma**2))

    # Mask beyond actual radius
    mask = jnp.abs(x) <= radius
    kernel = jnp.where(mask, kernel, 0.0)
    kernel = kernel / jnp.maximum(jnp.sum(kernel), 1e-12)

    # Reshape for axis
    kernel_shape = [1] * array.ndim
    kernel_shape[axis] = MAX_SIZE
    kernel = kernel.reshape(kernel_shape)

    # Pad array
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (MAX_RADIUS, MAX_RADIUS)
    padded = jnp.pad(array, pad_width, mode="symmetric")

    # Convolve
    result = convolve(padded, kernel, mode="same")

    # Crop
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(MAX_RADIUS, -MAX_RADIUS)

    return result[tuple(slices)]


def gaussian_filter_jax(array: jnp.ndarray, sigma: float, truncate: float = 4.0) -> jnp.ndarray:
    """JAX implementation using separable 1D filters (matches scipy approach).

    Applies 1D Gaussian filter along each axis sequentially.
    """

    # Apply 1D filter along axis 0, then axis 1
    result = gaussian_filter1d_jax(array, sigma=sigma, axis=0, truncate=truncate)
    result = gaussian_filter1d_jax(result, sigma=sigma, axis=1, truncate=truncate)
    return result


# TODO: potentially move this and other similar utils functions to an utils file
def rotate_kernel(kernel: jnp.ndarray, angle_deg: float, order: int = 1, mode: str = "constant") -> jnp.ndarray:
    """Rotate a kernel by a given angle."""

    # Convert angle to radians
    angle = jnp.deg2rad(angle_deg)

    # Get kernel shape
    ny, nx = kernel.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Create meshgrid of output coordinates
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")

    # Center coordinates around center
    y_centered = y - cy
    x_centered = x - cx

    # Apply inverse rotation to input coordinates
    x_src = x_centered * jnp.cos(angle) + y_centered * jnp.sin(angle) + cx
    y_src = -x_centered * jnp.sin(angle) + y_centered * jnp.cos(angle) + cy

    # Stack coordinates for map_coordinates
    coords = jnp.stack([y_src, x_src], axis=0)

    # Interpolate kernel
    rotated_kernel = map_coordinates(kernel, coords, order=order, mode=mode)

    return rotated_kernel


def apply_circular_mask(kernel: jnp.ndarray):
    """Apply a circular mask to a kernel."""

    # Get kernel shape
    ny, nx = kernel.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Create radius array
    y = jnp.arange(ny)[:, None]
    x = jnp.arange(nx)[None, :]
    r = jnp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rad = jnp.minimum(cy, cx) - 0.5

    # Create mask
    mask = (r <= rad).astype(kernel.dtype)
    masked_kernel = kernel * mask

    return masked_kernel