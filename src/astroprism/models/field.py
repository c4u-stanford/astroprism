"""
field.py

Latent spatial field model using NIFTy8 correlated fields.
"""

# === Imports ======================================================================================

from typing import Any

import jax
import jax.numpy as jnp
import nifty8.re as jft
from astroprism.utils.config import get_defaults

# === Main =========================================================================================

class FieldModel(jft.Model):
    """A collection of independent correlated field models for each channel."""

    def __init__(
        self,
        n_channels: int,
        shape: tuple[int, ...],
        distances: tuple[float, ...],
        offset_mean: float = None,
        offset_std: tuple[float, float] = None,
        fluctuations: tuple[float, float] = None,
        loglogavgslope: tuple[float, float] = None,
        flexibility: tuple[float, float] = None,
        asperity: tuple[float, float] | None = None,
        name: str = "",
        prefix: str = "",
    ):
        gp = get_defaults()["field"]
        offset_mean    = offset_mean    if offset_mean    is not None else gp["offset_mean"]
        offset_std     = offset_std     if offset_std     is not None else gp["offset_std"]
        fluctuations   = fluctuations   if fluctuations   is not None else gp["fluctuations"]
        loglogavgslope = loglogavgslope if loglogavgslope is not None else gp["loglogavgslope"]
        flexibility    = flexibility    if flexibility    is not None else gp["flexibility"]

        # Extract prior_params if config dicts are passed
        offset_std     = offset_std["prior_params"]     if isinstance(offset_std, dict)     else offset_std
        fluctuations   = fluctuations["prior_params"]   if isinstance(fluctuations, dict)   else fluctuations
        loglogavgslope = loglogavgslope["prior_params"] if isinstance(loglogavgslope, dict) else loglogavgslope
        flexibility    = flexibility["prior_params"]    if isinstance(flexibility, dict)    else flexibility

        correlated_fields = self._build_correlated_field(
            shape,
            distances,
            offset_mean,
            offset_std,
            fluctuations,
            loglogavgslope,
            flexibility,
            asperity,
            name=name,
            prefix=prefix,
        )
        domain = {
            k: jft.ShapeWithDtype((n_channels,) + correlated_fields.domain[k].shape, jnp.float64)
            for k in correlated_fields.domain.keys()
        }
        self._correlated_fields = correlated_fields
        self._n_channels = n_channels
        self._shape = shape
        self._distances = distances
        super().__init__(domain=domain, white_init=True)

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def distances(self) -> tuple[float, ...]:
        return self._distances

    def __call__(self, x: dict[str, Any]) -> jnp.ndarray:
        return jax.vmap(self._correlated_fields)(x)

    def sample(self, key: int | jax.Array) -> jnp.ndarray:
        """Generate a random sample from the prior."""
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        params = self.init(key)
        return jnp.array(self(params))

    @staticmethod
    def _build_correlated_field(
        shape,
        distances,
        offset_mean,
        offset_std,
        fluctuations,
        loglogavgslope,
        flexibility,
        asperity,
        name="",
        prefix="",
    ):
        """Create an instance of a correlated field model."""
        cf_zero_mode = dict(offset_mean=offset_mean, offset_std=offset_std)
        cf_fluctuations = dict(
            fluctuations=fluctuations,
            loglogavgslope=loglogavgslope,
            flexibility=flexibility,
            asperity=asperity,
        )
        cfm = jft.CorrelatedFieldMaker(name)
        cfm.set_amplitude_total_offset(**cf_zero_mode)
        cfm.add_fluctuations(
            shape, distances=distances, **cf_fluctuations, prefix=prefix, non_parametric_kind="amplitude"
        )
        return cfm.finalize()
