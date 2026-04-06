"""
config.py

Config loading with default fallback and deep merge.
"""

# === Imports ======================================================================================

import yaml
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"

# === Main =========================================================================================

def load_config(user_path: Optional[str | Path] = None) -> dict:
    """
    Load config, merging user overrides on top of defaults.

    Parameters
    ----------
    user_path : str or Path, optional
        Path to a user YAML config. Only the keys specified will override defaults.
        If None, returns the default config as-is.

    Returns
    -------
    dict

    Example
    -------
    cfg = load_config()                          # all defaults
    cfg = load_config("configs/my_run.yaml")     # defaults + overrides
    cfg["inference"]["output_directory"] = "..."  # further overrides in-place
    """
    cfg = yaml.safe_load(open(DEFAULT_CONFIG_PATH))

    if user_path is not None:
        user = yaml.safe_load(open(user_path))
        cfg = _deep_merge(cfg, user)

    return cfg


# === Utilities ====================================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.
    Override values take precedence. base is not mutated.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
