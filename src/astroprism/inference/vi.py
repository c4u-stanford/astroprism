"""
vi.py

Variational inference using NIFTy8's geometric VI (GeoVI/MGVI).
"""

# === Imports ======================================================================================

import jax
import jax.numpy as jnp
import nifty8.re as jft
from typing import Optional, Tuple, Any, Callable

# === Main =========================================================================================

class VariationalInference:
    """
    Wrapper for NIFTy8.re variational inference (MGVI/GeoVI).

    Supports parameter scheduling via the constants and point_estimates arguments to run():
    - constants: parameters frozen completely (not optimised)
    - point_estimates: parameters MAP-estimated rather than sampled

    Both are callables: iteration -> tuple[str].

    Example
    -------
    vi = VariationalInference(likelihood, ls_absdelta=1e-3, kl_absdelta=1e-2)
    samples, state = vi.run(
        n_iterations=20,
        n_samples=5,
        output_directory="output/run_001",
        constants=lambda i: ("background", "signal_gain") if i < 5 else (),
    )
    """

    def __init__(
        self,
        likelihood: jft.Model,
        *,
        seed: int = 42,
        sample_mode: str = "linear_resample",
        ls_absdelta: float = 1e-3,
        ls_maxiter: int = 1000,
        nls_xtol: float = 1e-3,
        nls_maxiter: int = 5,
        kl_absdelta: float = 1e-2,
        kl_xtol: float = 1e-8,
        kl_maxiter: int = 100,
    ):
        self.likelihood = likelihood
        self.sample_mode = sample_mode
        self.ls_absdelta = ls_absdelta
        self.ls_maxiter = ls_maxiter
        self.nls_xtol = nls_xtol
        self.nls_maxiter = nls_maxiter
        self.kl_absdelta = kl_absdelta
        self.kl_xtol = kl_xtol
        self.kl_maxiter = kl_maxiter

        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)
        self.init_params = {k: jnp.array(v, dtype=jnp.float64) for k, v in likelihood.init(init_key).items()}

    def run(
        self,
        n_iterations: int = 10,
        n_samples: int = 5,
        output_directory: str = "vi_results",
        resume: bool = True,
        constants: Optional[Callable[[int], tuple]] = None,
        point_estimates: Optional[Callable[[int], tuple]] = None,
    ) -> Tuple[Any, Any]:
        """
        Run the KL optimisation loop.

        Parameters
        ----------
        n_iterations : int
            Number of outer VI iterations.
        n_samples : int
            Number of samples per iteration for the KL estimate.
        output_directory : str
            Directory for checkpoints and results.
        resume : bool
            Resume from existing checkpoint if available. When resuming,
            NIFTy loads the saved state from output_directory and init_params
            is ignored.
        constants : callable(iteration) -> tuple[str], optional
            Parameters completely frozen at each iteration.
        point_estimates : callable(iteration) -> tuple[str], optional
            Parameters MAP-estimated (not sampled) at each iteration.

        Returns
        -------
        samples, state
        """
        self.key, opt_key = jax.random.split(self.key)

        draw_linear_kwargs = dict(
            cg_name="linear_solver",
            cg_kwargs=dict(absdelta=self.ls_absdelta, maxiter=self.ls_maxiter),
        )
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(
                name="nonlinear_solver",
                xtol=self.nls_xtol,
                cg_kwargs=dict(name=None),
                maxiter=self.nls_maxiter,
            )
        )
        kl_kwargs = dict(
            minimize=jft.optimize._newton_cg,
            minimize_kwargs=dict(
                name="kl_minimizer",
                xtol=self.kl_xtol,
                absdelta=self.kl_absdelta,
                cg_kwargs=dict(name=None),
                maxiter=self.kl_maxiter,
            ),
        )

        samples, state = jft.optimize_kl(
            self.likelihood,
            jft.Vector(self.init_params),
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            key=opt_key,
            odir=output_directory,
            resume=resume,
            sample_mode=self.sample_mode,
            draw_linear_kwargs=draw_linear_kwargs,
            nonlinearly_update_kwargs=nonlinearly_update_kwargs,
            kl_kwargs=kl_kwargs,
            constants=constants,
            point_estimates=point_estimates,
        )

        return samples, state


# === Utilities ====================================================================================

def run_inference(
    likelihood: jft.Model,
    n_iterations: int = 10,
    n_samples: int = 5,
    output_directory: str = "vi_results",
    seed: int = 42,
) -> Tuple[Any, Any]:
    """Convenience wrapper to run VI in one line with default settings."""
    vi = VariationalInference(likelihood, seed=seed)
    return vi.run(n_iterations, n_samples, output_directory)
