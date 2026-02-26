"""
dataset.py

Dataset classes for loading and managing multi-instrument astronomical data.
"""

# === Imports ======================================================================================

import glob
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from astroprism.io.instrument_loaders import _INSTRUMENT_LOADERS

# === Main =========================================================================================

class BaseDataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __getitem__(self, key: str | int) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
        """Get item from dataset by channel key (name) or integer index."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of channels in the dataset."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[jnp.ndarray, WCS, jnp.ndarray]]:
        """Iterate over the dataset."""
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of channels in the dataset."""
        pass

    @property
    @abstractmethod
    def shapes(self) -> list[tuple[int, int]]:
        """List of shapes (ny, nx) for each channel."""
        pass

    @property
    @abstractmethod
    def readout(self) -> list[jnp.ndarray]:
        """Readout mask for each channel (List of bool arrays)."""
        pass

    @property
    @abstractmethod
    def pixel_scales(self) -> list[tuple[float, float]]:
        """Get pixel scales (dy, dx) in arcsec for each channel."""
        pass

@dataclass
class SingleInstrumentDataset(BaseDataset):
    """Dataset for a single instrument."""

    data: list[jnp.ndarray]
    wcs: list[WCS]
    psfs: list[jnp.ndarray]
    channel_keys: list[str]
    masks: list[jnp.ndarray] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # Post-initialization
    def __post_init__(self):
        """Post-initialization checks and transformations."""
        n_channels = self.n_channels
        if n_channels != len(self.wcs) or n_channels != len(self.psfs) or n_channels != len(self.channel_keys):
            raise ValueError("Number of channels in the dataset must match the number of WCS, PSF, and keys.")

        if self.masks is not None and len(self.masks) != n_channels:
            raise ValueError("Number of masks must match number of channels.")

    # Special methods
    def __getitem__(self, key: str | int) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
        """Get item from dataset by channel key (name) or integer index."""
        # Access by string key
        if isinstance(key, str):
            try:
                index = self.channel_keys.index(key)
            except ValueError as err:
                raise KeyError(f"Channel key '{key}' not found in dataset keys.") from err

        # Access by integer index
        elif isinstance(key, int):
            if 0 <= key < self.n_channels:
                index = key
            else:
                raise IndexError(f"Index {key} out of range for {self.n_channels} channels.")
        else:
            raise TypeError(f"Dataset index must be a string (key) or an integer, not {type(key).__name__}.")

        data_channel = self.data[index]
        wcs_channel = self.wcs[index]
        psf_channel = self.psfs[index]

        return data_channel, wcs_channel, psf_channel

    def __len__(self) -> int:
        """Number of channels in the dataset."""
        return self.n_channels

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, WCS, jnp.ndarray]]:
        """Iterate over the dataset."""
        return iter(zip(self.data, self.wcs, self.psfs, strict=False))

    # Properties
    @property
    def n_channels(self) -> int:
        """Number of channels in the dataset."""
        return len(self.data)

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """List of shapes (ny, nx) for each channel."""
        return [d.shape for d in self.data]

    @property
    def pixel_scales(self) -> list[tuple[float, float]]:
        """Get pixel scales (dy, dx) in arcsec for each channel."""
        pixel_scales = []
        for w in self.wcs:
            scales_deg = proj_plane_pixel_scales(w)
            scales_arcsec = (float(scales_deg[1] * 3600), float(scales_deg[0] * 3600))
            pixel_scales.append(scales_arcsec)
        return pixel_scales

    @property
    def readout(self) -> list[jnp.ndarray]:
        """Readout mask for each channel (List of bool arrays)."""
        return [(jnp.isfinite(d) & (d > 0)).astype(bool) for d in self.data]

    def summary(self) -> str:
        """Summary of the dataset."""
        lines = []
        lines.append("SingleInstrumentDataset Summary:")
        lines.append("--------------------------------")
        lines.append(f"Number of channels: {self.n_channels}")
        lines.append(f"Channel keys: {self.channel_keys}")
        lines.append(f"Channel shapes: {self.shapes}")
        lines.append(f"Pixel scales: {self.pixel_scales}")
        return "\n".join(lines) + "\n"

@dataclass
class MultiInstrumentDataset(BaseDataset):
    """Dataset for multiple instruments."""

    instrument_datasets: dict[str, SingleInstrumentDataset]
    meta: dict[str, Any] = field(default_factory=dict)

    # Post-initialization
    def __post_init__(self):
        """Post-initialization checks."""
        if not self.instrument_datasets:
            raise ValueError("Must provide at least one instrument dataset")

        for instrument_key, ds in self.instrument_datasets.items():
            if not isinstance(ds, SingleInstrumentDataset):
                raise TypeError(
                    f"Instrument '{instrument_key}' must be a SingleInstrumentDataset, not {type(ds).__name__}"
                )

    # Special methods
    def __getitem__(self, key: str | int) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
        """Get item from dataset by channel key (name) or integer index."""
        # Access by string key
        if isinstance(key, str):
            try:
                index = self.channel_keys.index(key)
            except ValueError as err:
                raise KeyError(f"Channel key '{key}' not found in dataset keys.") from err

        # Access by integer index
        elif isinstance(key, int):
            if 0 <= key < self.n_channels:
                index = key
            else:
                raise IndexError(f"Index {key} out of range for {self.n_channels} channels.")
        else:
            raise TypeError(f"Dataset index must be a string (key) or an integer, not {type(key).__name__}.")

        data_channel = self.data[index]
        wcs_channel = self.wcs[index]
        psf_channel = self.psfs[index]

        return data_channel, wcs_channel, psf_channel

    def __len__(self) -> int:
        """Number of channels in the dataset."""
        return sum(len(ds) for ds in self.instrument_datasets.values())

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, WCS, jnp.ndarray]]:
        """Iterate over the dataset."""
        return iter(zip(self.data, self.wcs, self.psfs, strict=False))

    # Properties
    @property
    def data(self) -> list[jnp.ndarray]:
        """Data for each channel."""
        data = []
        for ds in self.instrument_datasets.values():
            data.extend(ds.data)
        return data

    @property
    def wcs(self) -> list[WCS]:
        """WCS for each channel."""
        wcs = []
        for ds in self.instrument_datasets.values():
            wcs.extend(ds.wcs)
        return wcs

    @property
    def psfs(self) -> list[jnp.ndarray]:
        """PSFs for each channel."""
        psfs = []
        for ds in self.instrument_datasets.values():
            psfs.extend(ds.psfs)
        return psfs

    @property
    def channel_keys(self) -> list[str]:
        """Channel keys for each channel."""
        keys = []
        for instrument_key, ds in self.instrument_datasets.items():
            for channel_key in ds.channel_keys:
                keys.append(f"{instrument_key}:{channel_key}")
        return keys

    @property
    def n_channels(self) -> int:
        """Number of channels in the dataset."""
        return sum(len(ds) for ds in self.instrument_datasets.values())

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """List of shapes (ny, nx) for each channel."""
        shapes = []
        for ds in self.instrument_datasets.values():
            shapes.extend(ds.shapes)
        return shapes

    @property
    def pixel_scales(self) -> list[tuple[float, float]]:
        """Get pixel scales (dy, dx) in arcsec for each channel."""
        pixel_scales = []
        for ds in self.instrument_datasets.values():
            pixel_scales.extend(ds.pixel_scales)
        return pixel_scales

    @property
    def readout(self) -> list[jnp.ndarray]:
        """Readout mask for each channel (List of bool arrays)."""
        readout = []
        for ds in self.instrument_datasets.values():
            readout.extend(ds.readout)
        return readout

    @property
    def n_instruments(self) -> int:
        """Number of instruments in the dataset."""
        return len(self.instrument_datasets)

    @property
    def instrument_keys(self) -> list[str]:
        """Instrument keys for each instrument."""
        return list(self.instrument_datasets.keys())

    def get_instrument_dataset(self, instrument_key: str) -> SingleInstrumentDataset:
        return self.instrument_datasets[instrument_key]

    def summary(self) -> str:
        """Summary of the dataset."""
        lines = []
        lines.append("MultiInstrumentDataset Summary:")
        lines.append("--------------------------------")
        lines.append(f"Number of instruments: {self.n_instruments}")
        lines.append(f"Instrument keys: {self.instrument_keys}")
        lines.append(f"Number of channels: {self.n_channels}")
        lines.append(f"Channel keys: {self.channel_keys}")
        lines.append(f"Channel shapes: {self.shapes}")
        lines.append(f"Pixel scales: {self.pixel_scales}")

        return "\n".join(lines) + "\n"

def load_dataset(path: str, instrument: str, extension: str = "fits") -> SingleInstrumentDataset:
    # Get instrument loader
    loader = _INSTRUMENT_LOADERS[instrument]

    # Determine file pattern
    if os.path.isdir(path):
        pattern = os.path.join(path, f"*.{extension}")
    else:
        pattern = path

    # Get files
    files = sorted(glob.glob(pattern))
    channel_keys = [os.path.basename(f).split(".")[0] for f in files]

    # Load data
    data_list = []
    wcs_list = []
    psf_list = []

    for file in files:
        if file.endswith(extension):
            data, wcs, psf = loader(file)
            data_list.append(data)
            wcs_list.append(wcs)
            psf_list.append(psf)

    return SingleInstrumentDataset(data=data_list, wcs=wcs_list, psfs=psf_list, channel_keys=channel_keys)

def load_multiple_datasets(paths_dict: dict[str, str], extension: str = "fits") -> MultiInstrumentDataset:
    """Load multiple datasets from a dictionary of paths."""
    instrument_datasets = {}
    for instrument, path in paths_dict.items():
        instrument_datasets[instrument] = load_dataset(path=path, instrument=instrument, extension=extension)
    return MultiInstrumentDataset(instrument_datasets=instrument_datasets)

    