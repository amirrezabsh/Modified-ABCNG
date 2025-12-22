from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_env(path: str = ".env") -> Dict[str, str]:
    env: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return env
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip()
    return env


def _coerce_value(val: str):
    low = val.lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def get_abcng_kwargs(env: Dict[str, str]) -> Dict[str, object]:
    mapping = {
        "ABCNG_USE_GBEST": "use_gbest",
        "ABCNG_USE_ADAPTIVE_K": "use_adaptive_k",
        "ABCNG_USE_GAUSSIAN": "use_gaussian",
        "ABCNG_NEIGHBOR_MODE": "neighbor_mode",
        "ABCNG_NOISE_MODEL": "noise_model",
        "ABCNG_K_FIXED": "k_fixed",
        "ABCNG_UPDATE_DIM_MODE": "update_dim_mode",
    }
    kwargs: Dict[str, object] = {}
    for env_key, kw_key in mapping.items():
        if env_key in env:
            kwargs[kw_key] = _coerce_value(env[env_key])
    return kwargs
