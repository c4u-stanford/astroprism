"""
masking.py

Masking utilities for held-out validation.

Strategy syntax
---------------
- null / None          → no masking
- "full"               → mask entire channel
- 0.3                  → random 30% of all pixels
- "bottom"             → mask entire bottom half
- "bottom:0.3"         → random 30% within bottom half
- "top_left:0.5"       → random 50% within top-left quadrant

Convention: True = kept for training, False = held out for test.
"""

# === Imports ======================================================================================

import numpy as np

# === Main =========================================================================================

REGIONS = {"full", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"}


def build_masks(
    shapes: list[tuple[int, int]],
    strategies: list,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Build per-channel masks from strategies.

    Parameters
    ----------
    shapes : list of (ny, nx)
        Shape of each channel.
    strategies : list
        One entry per channel. See module docstring for syntax.
    seed : int
        Random seed for reproducible random masking.

    Returns
    -------
    list of np.ndarray (bool)
        One mask per channel. True = training, False = held out.
    """
    if len(shapes) != len(strategies):
        raise ValueError(
            f"Number of strategies ({len(strategies)}) must match "
            f"number of channels ({len(shapes)})."
        )

    rng = np.random.default_rng(seed)
    masks = []
    for shape, strategy in zip(shapes, strategies):
        masks.append(_build_channel_mask(shape, strategy, rng))
    return masks


def save_masks(masks: list[np.ndarray], path: str) -> None:
    """Save masks to a .npz file."""
    np.savez(path, *[np.asarray(m) for m in masks])


def load_masks(path: str) -> list[np.ndarray]:
    """Load masks from a .npz file."""
    data = np.load(path)
    return [data[k] for k in sorted(data.files)]


# === Utilities ====================================================================================

def _build_channel_mask(
    shape: tuple[int, int],
    strategy,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a single-channel mask from a strategy spec."""
    # No masking
    if strategy is None:
        return np.ones(shape, dtype=bool)

    # Pure float → random fraction over full image
    if isinstance(strategy, (int, float)):
        return _random_mask(shape, float(strategy), rng)

    # String parsing
    strategy = str(strategy).strip().lower()
    if strategy in ("", "none"):
        return np.ones(shape, dtype=bool)

    # Check for "region:fraction" syntax
    if ":" in strategy:
        region, frac_str = strategy.split(":", 1)
        region = region.strip()
        fraction = float(frac_str.strip())
    else:
        region = strategy
        fraction = 1.0

    if region not in REGIONS:
        raise ValueError(f"Unknown region: '{region}'. Choose from {sorted(REGIONS)}")

    # Build region mask (True = pixel is in the region)
    region_mask = _region_mask(shape, region)

    # Apply fraction within that region
    if fraction >= 1.0:
        # Mask the entire region
        keep = ~region_mask
    elif fraction <= 0.0:
        # Mask nothing
        keep = np.ones(shape, dtype=bool)
    else:
        # Random fraction within the region
        keep = np.ones(shape, dtype=bool)
        region_indices = np.argwhere(region_mask)
        n_mask = int(len(region_indices) * fraction)
        chosen = rng.choice(len(region_indices), size=n_mask, replace=False)
        for idx in region_indices[chosen]:
            keep[idx[0], idx[1]] = False

    return keep


def _region_mask(shape: tuple[int, int], region: str) -> np.ndarray:
    """Return a boolean array where True = pixel belongs to the named region."""
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    mask = np.zeros(shape, dtype=bool)

    if region == "full":
        mask[:] = True
    elif region == "left":
        mask[:, :cx] = True
    elif region == "right":
        mask[:, cx:] = True
    elif region == "top":
        mask[cy:, :] = True
    elif region == "bottom":
        mask[:cy, :] = True
    elif region == "top_left":
        mask[cy:, :cx] = True
    elif region == "top_right":
        mask[cy:, cx:] = True
    elif region == "bottom_left":
        mask[:cy, :cx] = True
    elif region == "bottom_right":
        mask[:cy, cx:] = True

    return mask


def _random_mask(shape: tuple[int, int], fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Mask a random fraction of all pixels. Shortcut for region='full'."""
    if fraction >= 1.0:
        return np.zeros(shape, dtype=bool)
    if fraction <= 0.0:
        return np.ones(shape, dtype=bool)

    keep = np.ones(shape, dtype=bool)
    n_pixels = shape[0] * shape[1]
    n_mask = int(n_pixels * fraction)
    flat_indices = rng.choice(n_pixels, size=n_mask, replace=False)
    keep.ravel()[flat_indices] = False
    return keep
