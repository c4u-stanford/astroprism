"""
results.py

Load and analyze results from a completed astroprism inference run.
"""

# === Imports ======================================================================================

import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from astroprism.models.field import FieldModel
from astroprism.models.signal import SignalModel
from astroprism.models.response import InstrumentResponse
from astroprism.models.noise import NoiseModel
from astroprism.models.forward import ForwardModel
from astroprism.io.dataset import load_dataset

# === Main =========================================================================================

class PosteriorResult:
    """
    Load and analyze results from a completed astroprism run.

    Parameters
    ----------
    run_dir : str or Path
        Path to the output directory containing config.yaml and last.pkl.

    Examples
    --------
    result = PosteriorResult("output/jwst_miri_tutorial")
    signal_mean = result.signal_mean()       # (n_channels, ny, nx)
    signal_std = result.signal_std()         # (n_channels, ny, nx)
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)

        with open(self.run_dir / "config.yaml") as f:
            self._config = yaml.safe_load(f)

        files_path = self.run_dir / "files_used.yaml"
        if files_path.exists():
            with open(files_path) as f:
                self._files_used = yaml.safe_load(f)
        else:
            self._files_used = None

        self._samples = None
        self._state = None
        self._dataset = None

    @property
    def config(self) -> dict:
        """The full config used for this run."""
        return self._config

    @property
    def derived(self) -> dict:
        """Derived values (n_channels, signal_shape, distances, etc.)."""
        return self._config["_derived"]

    # === Loading =================================================================================

    def load_samples(self, filename: str = "last.pkl"):
        """Load posterior samples from checkpoint."""
        if self._samples is None:
            with open(self.run_dir / filename, "rb") as f:
                self._samples, self._state = pickle.load(f)
        return self._samples, self._state

    def load_dataset(self):
        """Re-load the dataset from saved file paths."""
        if self._dataset is None:
            if self._files_used is None:
                raise FileNotFoundError("No files_used.yaml found — cannot reload dataset.")
            self._dataset = load_dataset(
                path=self._files_used["data_path"],
                instrument=self._files_used["instrument"],
                extension=self._files_used.get("extension", "fits"),
            )
        return self._dataset

    # === Model reconstruction ====================================================================

    def build_signal_model(self) -> SignalModel:
        """Reconstruct the signal model from saved config (no dataset needed)."""
        d = self.derived
        spatial_gp = FieldModel(
            n_channels=d["n_channels"],
            shape=tuple(d["signal_shape"]),
            distances=tuple(d["distances"]),
        )
        return SignalModel(spatial_gp)

    def build_response_model(self, dataset=None) -> InstrumentResponse:
        """Reconstruct the instrument response model (needs dataset for WCS/PSFs)."""
        if dataset is None:
            dataset = self.load_dataset()
        d = self.derived
        return InstrumentResponse(
            dataset=dataset,
            signal_wcs=dataset.wcs[d["ref_idx"]],
            signal_shape=tuple(d["signal_shape"]),
        )

    def build_noise_model(self) -> NoiseModel:
        """Reconstruct the noise model from saved config."""
        return NoiseModel(n_channels=self.derived["n_channels"])

    def build_forward_model(self, dataset=None) -> ForwardModel:
        """Reconstruct the full forward model."""
        return ForwardModel(
            signal_model=self.build_signal_model(),
            response_model=self.build_response_model(dataset),
            noise_model=self.build_noise_model(),
        )

    # === Predictions =============================================================================

    def predict_signal(self, samples=None) -> list[jnp.ndarray]:
        """
        Compute the posterior signal (sky reconstruction) for each sample.

        Returns list of arrays, each with shape (n_channels, ny, nx).
        """
        if samples is None:
            samples, _ = self.load_samples()
        gp = self.build_signal_model()
        return [jnp.array(gp(s.tree)) for s in samples]

    def predict_response(self, samples=None, dataset=None) -> list[list[jnp.ndarray]]:
        """
        Compute the instrument response (model-predicted data) for each sample.

        Returns list of lists — outer: per sample, inner: per channel.
        """
        if samples is None:
            samples, _ = self.load_samples()
        gp = self.build_signal_model()
        response = self.build_response_model(dataset)
        results = []
        for s in samples:
            x = s.tree
            signal = gp(x)
            results.append(response(x, signal))
        return results

    def predict_noise_std(self, samples=None, dataset=None) -> list[list[jnp.ndarray]]:
        """
        Compute the noise standard deviation for each sample.

        Returns list of lists — outer: per sample, inner: per channel.
        """
        if samples is None:
            samples, _ = self.load_samples()
        gp = self.build_signal_model()
        response = self.build_response_model(dataset)
        noise = self.build_noise_model()
        results = []
        for s in samples:
            x = s.tree
            signal = gp(x)
            resp = response(x, signal)
            results.append(noise(x, resp))
        return results

    def signal_mean(self) -> jnp.ndarray:
        """Posterior mean of the signal. Shape: (n_channels, ny, nx)."""
        signals = self.predict_signal()
        return jnp.mean(jnp.stack(signals), axis=0)

    def signal_std(self) -> jnp.ndarray:
        """Posterior std of the signal. Shape: (n_channels, ny, nx)."""
        signals = self.predict_signal()
        return jnp.std(jnp.stack(signals), axis=0)
