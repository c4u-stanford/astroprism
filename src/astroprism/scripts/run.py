"""
run.py

CLI entry point for running the astroprism inference pipeline.

Usage
-----
astroprism-run --config configs/my_run.yaml
"""

# === Imports ======================================================================================

# Must be set before any JAX/NIFTy import
import jax
jax.config.update("jax_enable_x64", True)

import argparse
import os
import sys
from pathlib import Path

import yaml

from astroprism.utils import load_config
from astroprism.utils.masking import build_masks, save_masks
from astroprism.io.dataset import load_dataset
from astroprism.models.gp import SpatialGP, MixtureGP
from astroprism.models.response import InstrumentResponse
from astroprism.models.noise import NoiseModel
from astroprism.models.forward import ForwardModel
from astroprism.models.likelihood import build_likelihood
from astroprism.inference.vi import VariationalInference

# === Main =========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run astroprism Bayesian inference pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, metavar="PATH", help="Path to config YAML")
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)

    # 2. Load dataset
    data_cfg = cfg["data"]
    if data_cfg["path"] is None:
        print("Error: data.path must be set in your config file.", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(
        path=data_cfg["path"],
        instrument=data_cfg["instrument"],
        extension=data_cfg.get("extension", "fits"),
    )
    print(dataset.summary())

    # 3. Signal grid — use finest-resolution channel as reference
    pixel_scales = dataset.pixel_scales
    ref_idx = min(range(dataset.n_channels), key=lambda i: pixel_scales[i][0])
    signal_wcs   = dataset.wcs[ref_idx]
    signal_shape = dataset.shapes[ref_idx]
    distances    = pixel_scales[ref_idx]

    # 4. Build models
    spatial_gp = SpatialGP(
        n_channels=dataset.n_channels,
        shape=signal_shape,
        distances=distances,
    )
    gp_model = MixtureGP(spatial_gp)

    response_model = InstrumentResponse(
        dataset=dataset,
        signal_wcs=signal_wcs,
        signal_shape=signal_shape,
    )
    noise_model   = NoiseModel(n_channels=dataset.n_channels)
    forward_model = ForwardModel(
        gp_model=gp_model,
        response_model=response_model,
        noise_model=noise_model,
    )

    # 5. Build training mask
    mask = dataset.readout
    masking_cfg = cfg.get("masking", {})
    holdout = None
    if masking_cfg.get("enabled", False):
        strategies = masking_cfg["strategies"]
        holdout = build_masks(dataset.shapes, strategies, seed=masking_cfg.get("seed", 42))
        mask = [r & h for r, h in zip(mask, holdout)]

    # 6. Save config and artifacts to output directory
    output_dir = cfg["inference"]["output_directory"]
    _save_run_artifacts(output_dir, cfg, dataset, signal_shape, distances, ref_idx, holdout)

    # 7. Build likelihood
    likelihood = build_likelihood(dataset, forward_model, mask=mask)

    # 8. Build parameter schedule from config
    constants, point_estimates = _build_schedule(cfg["params"])

    # 9. Run inference
    inf_cfg = cfg["inference"]
    slv_cfg = cfg["solver"]

    vi = VariationalInference(
        likelihood,
        seed=inf_cfg["seed"],
        sample_mode=inf_cfg["sample_mode"],
        **slv_cfg,
    )

    samples, state = vi.run(
        n_iterations=inf_cfg["n_iterations"],
        n_samples=inf_cfg["n_samples"],
        output_directory=inf_cfg["output_directory"],
        resume=inf_cfg["resume"],
        constants=constants,
        point_estimates=point_estimates,
    )

    print(f"Done. Results saved to: {inf_cfg['output_directory']}")
    return samples, state


# === Utilities ====================================================================================

def _save_run_artifacts(output_dir, cfg, dataset, signal_shape, distances, ref_idx, holdout):
    """Save config, derived values, and masks to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Augment config with derived values
    cfg["_derived"] = {
        "n_channels": dataset.n_channels,
        "signal_shape": list(signal_shape),
        "distances": list(distances),
        "ref_idx": ref_idx,
        "channel_keys": dataset.channel_keys,
        "shapes": [list(s) for s in dataset.shapes],
        "pixel_scales": [list(ps) for ps in dataset.pixel_scales],
    }
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Save data file paths for reproducibility
    data_cfg = cfg["data"]
    files_used = {
        "data_path": str(Path(data_cfg["path"]).resolve()),
        "instrument": data_cfg["instrument"],
        "extension": data_cfg.get("extension", "fits"),
    }
    with open(os.path.join(output_dir, "files_used.yaml"), "w") as f:
        yaml.safe_dump(files_used, f, sort_keys=False)

    # Save holdout mask if masking is enabled
    if holdout is not None:
        save_masks(holdout, os.path.join(output_dir, "mask.npz"))


def _build_schedule(params_cfg: dict):
    """
    Build constants and point_estimates callables from the params config section.

    Each param entry can have:
      mode: sample | point_estimate | constant
      freeze_until: N   (frozen as constant for iters 0..N-1, then transitions to mode)
    """

    def constants(i: int) -> tuple:
        frozen = []
        for name, p in params_cfg.items():
            mode         = p.get("mode", "sample")
            freeze_until = p.get("freeze_until", 0)
            if mode == "constant" or (freeze_until > 0 and i < freeze_until):
                frozen.append(name)
        return tuple(frozen)

    def point_estimates(i: int) -> tuple:
        pe = []
        for name, p in params_cfg.items():
            mode         = p.get("mode", "sample")
            freeze_until = p.get("freeze_until", 0)
            # Only active as point_estimate once past freeze_until
            if mode == "point_estimate" and i >= freeze_until:
                pe.append(name)
        return tuple(pe)

    return constants, point_estimates
