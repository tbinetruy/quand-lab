from __future__ import annotations

from math import sqrt

import numpy as np

from quant_lab.domain import FloatArray


def brownian_increments(
    *,
    n_paths: int,
    n_steps: int,
    dt: float,
    seed: int | None = None,
    antithetic: bool = False,
) -> FloatArray:
    """Generate Brownian increments shaped as paths by time steps."""

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative when provided")

    rng = np.random.default_rng(seed)

    if antithetic:
        half_path_count = (n_paths + 1) // 2
        base_normals = rng.standard_normal(size=(half_path_count, n_steps))
        standard_normals = np.vstack((base_normals, -base_normals))[:n_paths]
    else:
        standard_normals = rng.standard_normal(size=(n_paths, n_steps))

    return np.asarray(sqrt(dt) * standard_normals, dtype=np.float64)

