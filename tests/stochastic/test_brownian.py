import numpy as np
import pytest

from quant_lab.stochastic import brownian_increments


def test_brownian_increments_are_reproducible_with_seed() -> None:
    first = brownian_increments(n_paths=3, n_steps=4, dt=0.1, seed=42)
    second = brownian_increments(n_paths=3, n_steps=4, dt=0.1, seed=42)

    np.testing.assert_array_equal(first, second)


def test_brownian_increments_have_expected_shape() -> None:
    increments = brownian_increments(n_paths=5, n_steps=7, dt=0.25, seed=1)

    assert increments.shape == (5, 7)


def test_brownian_increments_support_antithetic_variates() -> None:
    increments = brownian_increments(
        n_paths=4,
        n_steps=3,
        dt=1.0,
        seed=7,
        antithetic=True,
    )

    np.testing.assert_allclose(increments[0], -increments[2])
    np.testing.assert_allclose(increments[1], -increments[3])


def test_brownian_increments_reject_non_positive_paths() -> None:
    with pytest.raises(ValueError, match="n_paths must be positive"):
        brownian_increments(n_paths=0, n_steps=1, dt=0.1)


def test_brownian_increments_reject_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="n_steps must be positive"):
        brownian_increments(n_paths=1, n_steps=0, dt=0.1)


def test_brownian_increments_reject_negative_dt() -> None:
    with pytest.raises(ValueError, match="dt must be non-negative"):
        brownian_increments(n_paths=1, n_steps=1, dt=-0.1)


def test_brownian_increments_reject_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative when provided"):
        brownian_increments(n_paths=1, n_steps=1, dt=0.1, seed=-1)
