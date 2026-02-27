"""
likelihood.py

Gaussian likelihood construction for Bayesian inference.
"""

# === Setup ========================================================================================

import jax.numpy as jnp
import nifty8.re as jft

from astroprism.io import BaseDataset

# === Main =========================================================================================

class LikelihoodModel(jft.Model):
    """
    Wraps ForwardModel to apply masking and flattening for the Likelihood.
    """
    def __init__(self, forward_model, mask):
        self.forward_model = forward_model
        self.mask = mask
        
        # Pre-calculate indices to make JIT happy
        self.valid_indices_list = []
        total_pixels = 0
        
        for m in mask:
            # Note: m should be a concrete array (numpy/jax) here
            # ravel() ensures we get flat indices
            indices = jnp.where(m.ravel())[0]
            self.valid_indices_list.append(indices)
            total_pixels += len(indices)

        # Explicitly define target shape to avoid NonConcreteBooleanIndexError
        target = jft.ShapeWithDtype((total_pixels,), jnp.float64)
        
        super().__init__(
            domain=forward_model.domain, 
            init=forward_model.init,
            target=target
        )

    def __call__(self, x):
        mean, std = self.forward_model(x)
        
        # Use integer indexing instead of boolean masking
        mean_flat = jnp.concatenate([m.ravel()[idx] for m, idx in zip(mean, self.valid_indices_list)])
        std_flat = jnp.concatenate([s.ravel()[idx] for s, idx in zip(std, self.valid_indices_list)])

        
        return mean_flat, std_flat


def build_likelihood(dataset, observation_model, mask):
    # 1. Wrap the physics model for the likelihood
    likelihood_model = LikelihoodModel(observation_model, mask)
    
    # 2. Prepare the data (same masking logic)
    data_flat = jnp.concatenate([d.ravel()[m.ravel()] for d, m in zip(dataset.data, mask)])
    
    # 3. Build NIFTy Likelihood
    return jft.VariableCovarianceGaussian(data_flat).amend(likelihood_model)


# def build_likelihood(dataset: BaseDataset, model: jft.Model, mask: list[jnp.ndarray] | None = None) -> jft.Likelihood:
#     """
#     Construct the Gaussian Likelihood for the AstroPFM model.

#     Args:
#         model: The initialized ObservationModel (ObservationModel).
#                Must output (mean_flat, std_flat).
#         dataset: The dataset containing observed data.
#         mask: List of boolean masks (one per channel) indicating valid pixels.
#               Must match the mask used inside ObservationModel.

#     Returns:
#         likelihood: A NIFTy Likelihood object (VariableCovarianceGaussian).
#     """
#     # 1. Flatten the Observed Data
#     #    This must exactly match the flattening logic inside ObservationModel.__call__
#     if mask is not None:
#         # Concatenate valid pixels from all channels
#         data_flat = jnp.concatenate([d.ravel()[m.ravel()] for d, m in zip(dataset.data, mask, strict=False)])
#     else:
#         # Concatenate all pixels
#         data_flat = jnp.concatenate([d.ravel() for d in dataset.data])

#     # 2. Instantiate the Likelihood
#     #    NIFTy's VariableCovarianceGaussian compares 'data_flat' against
#     #    the (mean, std) tuple returned by 'model(x)'.
#     likelihood = jft.VariableCovarianceGaussian(data_flat).amend(model)

#     return likelihood
