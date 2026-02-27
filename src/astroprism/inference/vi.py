"""
vi.py

Variational inference using NIFTy8's geometric VI (GeoVI/MGVI).
"""

# === Imports ======================================================================================

import jax
import jax.numpy as jnp
import nifty8.re as jft
from typing import Optional, Tuple, Dict, Any

# === Main =========================================================================================

class VariationalInference:
    """
    Wrapper for NIFTy8.re Variational Inference (MGVI/GeoVI).
    """
    
    def __init__(
        self, 
        likelihood: jft.Model, 
        seed: int = 42
    ):
        """
        Args:
            likelihood: The energy model (negative log-likelihood).
            seed: Random seed for initialization and sampling.
        """
        self.likelihood = likelihood
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        
        # Initialize parameters using the model's init method
        # This creates the dictionary of parameters with correct shapes
        self.key, init_key = jax.random.split(self.key)
        self.init_params = likelihood.init(init_key)

    def run(
        self, 
        n_iterations: int = 10, 
        n_samples: int = 6, 
        output_directory: str = "vi_results",
        resume: bool = False,
        verbosity: int = 0
    ) -> Tuple[Any, Any]:
        """
        Run the KL optimization loop.
        
        Args:
            n_iterations: Number of global iterations (linear + non-linear updates).
            n_samples: Number of samples for the KL estimate (GeoVI samples).
            output_directory: Path to save results/checkpoints.
            resume: If True, try to resume from output_directory.
            verbosity: 0 for silent, higher for debug info.
            
        Returns:
            samples: The final samples from the approximate posterior.
            state: The final optimization state (contains mean, etc).
        """
        self.key, opt_key = jax.random.split(self.key)
        
        # Standard solver settings for robust convergence
        # Linear solver (CG) for drawing samples
        draw_linear_kwargs = dict(
            cg_name="linear_solver", 
            cg_kwargs=dict(absdelta=1e-4, maxiter=100)
        )
        
        # Non-linear solver for maximizing likelihood wrt latent parameters
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(
                name="nonlinear_solver", 
                xtol=1e-4, 
                cg_kwargs=dict(name=None), 
                maxiter=10
            )
        )
        
        # KL minimizer (Newton-CG) for updating the mean
        kl_kwargs = dict(
            minimize=jft.optimize._newton_cg,
            minimize_kwargs=dict(
                name="kl_minimizer", 
                xtol=1e-4, 
                absdelta=1e-4, 
                cg_kwargs=dict(name=None), 
                maxiter=10
            ),
        )

        if verbosity > 0:
            print(f"Starting VI optimization: {n_iterations} iterations, {n_samples} samples.")

        # Run NIFTy's optimize_kl
        samples, state = jft.optimize_kl(
            self.likelihood,
            jft.Vector(self.init_params),
            n_total_iterations=n_iterations,
            n_samples=n_samples,
            key=opt_key,
            odir=output_directory,
            resume=resume,
            draw_linear_kwargs=draw_linear_kwargs,
            nonlinearly_update_kwargs=nonlinearly_update_kwargs,
            kl_kwargs=kl_kwargs
        )
        
        return samples, state


def run_inference(
    likelihood: jft.Model, 
    n_iterations: int = 10, 
    n_samples: int = 6, 
    output_directory: str = "vi_results",
    seed: int = 42,
    verbosity: int = 1
):
    """Helper function to run VI in one line."""
    vi = VariationalInference(likelihood, seed=seed)
    samples, state = vi.run(n_iterations, n_samples, output_directory, verbosity=verbosity)
    return samples, state
