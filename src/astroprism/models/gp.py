"""
gp.py

Gaussian Process models for multi-channel astronomical imaging.
"""

# === Imports ======================================================================================

from typing import Any

import jax
import jax.numpy as jnp
import nifty8.re as jft
from astroprism.models.priors import build_prior

# === Main =========================================================================================

class SpatialGP(jft.Model):  # NOTE: other name may be 'GaussianRandomFields'
    """A collection of independent correlated field models for each channel"""

    def __init__(
        self,
        n_channels: int,
        shape: tuple[int, ...],  # NOTE: Tuple[float, float] should be used for 2D images
        distances: tuple[float, ...],
        offset_mean: float = 0.0,  # NOTE: single float for now, does this need a tuple?
        offset_std: tuple[float, float] = (1.0, 0.1),
        fluctuations: tuple[float, float] = (1.0, 0.1),
        loglogavgslope: tuple[float, float] = (-3.0, 1.0),
        flexibility: tuple[float, float] = (0.5, 0.1),
        asperity: tuple[float, float] | None = None,
        name: str = "",
        prefix: str = "",
    ):
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
        """Number of channels in the model."""
        return self._n_channels

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the model."""
        return self._shape

    @property
    def distances(self) -> tuple[float, ...]:
        """Distances between channels in the model."""
        return self._distances

    def __call__(self, x: dict[str, Any]) -> jnp.ndarray:
        return jax.vmap(self._correlated_fields)(x)

    # TODO: don't really need this, can just say self.init(key) and then self(params)
    def sample(self, key: int | jax.Array) -> jnp.ndarray:
        """Generate a random sample from the prior."""
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)  # NOTE: for production, provide the jax.random.PRNGKey directly

        params = self.init(key)
        sample = jnp.array(self(params))
        return sample

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
        """Create an instance of a correlated field model"""
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
        correlated_field = cfm.finalize()
        return correlated_field


class MixtureGP(jft.Model):
    """A mixture of Gaussian processes with a global mixing matrix"""

    # NOTE: technically latent fields not SpatialGP (maybe call it LatentGP?), from "Linear Mixing Model" ($s = A u$).
    # NOTE: Observed Signal ($s$): What you actually see in the sky (the mixed channels).
    # NOTE: Latent Signal ($u$): The hidden, underlying structure that generates the observations.

    # Supported activation functions
    ACTIVATIONS = {
        "exp": jnp.exp,
        "softplus": lambda x: jnp.log1p(jnp.exp(x)),
        "sigmoid": jax.nn.sigmoid,
        "identity": lambda x: x,
        "relu": jax.nn.relu,
        "square": lambda x: x**2,
    }

    def __init__(
        self,
        spatial_gps: SpatialGP,
        mix_mode: str = "full",
        mix_diag: dict = None,
        mix_off_diag: dict = None,
        mix_full: dict = None,
        mix_offset: dict = None,
        activation: str = "exp",
    ):
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(self.ACTIVATIONS.keys())}")
        self.activation = self.ACTIVATIONS[activation]
        self.activation_name = activation
        self.spatial_gps = spatial_gps
        self.mix_mode = mix_mode
        domain = self.spatial_gps.domain
        init = self.spatial_gps.init
        n_channels = self.spatial_gps.n_channels

        if self.mix_mode == "full":
            domain, init = self._build_full_mixing_matrix(n_channels, domain, init, mix_full)
        elif self.mix_mode == "diag":
            domain, init = self._build_diag_mixing_matrix(n_channels, domain, init, mix_diag, mix_off_diag)
        else:
            raise ValueError(f"Unknown mix_mode: {mix_mode}")

        domain, init = self._build_mixing_offset(n_channels, domain, init, mix_offset)
        super().__init__(domain=domain, init=init)

    @property
    def n_channels(self) -> int:
        """Number of channels in the model."""
        return self.spatial_gps.n_channels

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the model."""
        return self.spatial_gps.shape

    @property
    def distances(self) -> tuple[float, ...]:
        """Distances between channels in the model."""
        return self.spatial_gps.distances

    def __call__(self, x: dict[str, Any]) -> jnp.ndarray:
        base_outputs = self.spatial_gps({k: x[k] for k in self.spatial_gps.domain.keys()})

        if self.mix_mode == "full":
            mixing_matrix = self.mixing_matrix(x)
        elif self.mix_mode == "diag":
            diag = self.mixing_diag(x)
            off_diag = self.mixing_off_diag(x)
            mixing_matrix = _assemble_cholesky_matrix(diag, off_diag)

        mixture_output = jnp.tensordot(mixing_matrix, base_outputs, axes=(1, 0))
        return self.activation(mixture_output + self.mixing_offset(x)[:, None, None])

    def sample(self, key: int | jax.Array) -> jnp.ndarray:
        """Generate a random sample from the prior."""
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        params = self.init(key)
        sample = jnp.array(self(params))
        return sample

    def _build_full_mixing_matrix(self, n_channels, domain, init, mix_full):
        self.mixing_matrix = build_prior("mixture_matrix", mix_full, (n_channels, n_channels))
        return domain | self.mixing_matrix.domain, init | self.mixing_matrix.init

    def _build_diag_mixing_matrix(self, n_channels, domain, init, mix_diag, mix_off_diag):
        self.mixing_diag     = build_prior("mixing_diag",     mix_diag,     (n_channels,))
        self.mixing_off_diag = build_prior("mixing_off_diag", mix_off_diag, (_n_triangular_lower(n_channels),))
        return (
            domain | self.mixing_diag.domain | self.mixing_off_diag.domain,
            init | self.mixing_diag.init | self.mixing_off_diag.init,
        )

    def _build_mixing_offset(self, n_channels, domain, init, mix_offset):
        self.mixing_offset = build_prior("mixing_offset", mix_offset, (n_channels,))
        return domain | self.mixing_offset.domain, init | self.mixing_offset.init


# === Utilities ====================================================================================

def _assemble_cholesky_matrix(diag, off_diag):
    """Assemble a Cholesky matrix from a diagonal and off-diagonal vector"""
    n_channels = len(diag)
    L = jnp.zeros((n_channels, n_channels))
    L = L.at[jnp.diag_indices(n_channels)].set(1.0)
    tril_indices = jnp.tril_indices(n_channels, k=-1)
    L = L.at[tril_indices].set(off_diag)
    L = L * (jnp.exp(diag))[:, None]
    return L


def _n_triangular_lower(n):
    """Number of elements in the lower triangular part of a square matrix"""
    return (n * (n - 1)) // 2
