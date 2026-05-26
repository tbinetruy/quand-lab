import numpy as np
import pytest

from quant_lab.domain import MonteCarloConfig, MonteCarloResult


def test_monte_carlo_config_accepts_valid_inputs() -> None:
    config = MonteCarloConfig(
        n_paths=10_000,
        n_steps=252,
        seed=42,
        antithetic=True,
        return_paths=False,
    )

    assert config.n_paths == 10_000
    assert config.n_steps == 252
    assert config.seed == 42
    assert config.antithetic is True
    assert config.return_paths is False


def test_monte_carlo_config_rejects_non_positive_paths() -> None:
    with pytest.raises(ValueError, match="n_paths must be positive"):
        MonteCarloConfig(n_paths=0, n_steps=10)


def test_monte_carlo_config_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="n_steps must be positive"):
        MonteCarloConfig(n_paths=10, n_steps=0)


def test_monte_carlo_config_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative when provided"):
        MonteCarloConfig(n_paths=10, n_steps=10, seed=-1)


def test_monte_carlo_result_accepts_valid_arrays() -> None:
    result = MonteCarloResult(
        price=10.0,
        standard_error=0.1,
        confidence_interval_95=(9.8, 10.2),
        terminal_prices=np.array([95.0, 105.0], dtype=np.float64),
        paths=np.array([[100.0, 95.0], [100.0, 105.0]], dtype=np.float64),
    )

    assert result.price == 10.0
    assert result.terminal_prices.shape == (2,)
    assert result.paths is not None
    assert result.paths.shape == (2, 2)


def test_monte_carlo_result_rejects_negative_standard_error() -> None:
    with pytest.raises(ValueError, match="standard_error must be non-negative"):
        MonteCarloResult(
            price=10.0,
            standard_error=-0.1,
            confidence_interval_95=(9.8, 10.2),
            terminal_prices=np.array([100.0], dtype=np.float64),
        )


def test_monte_carlo_result_rejects_inverted_confidence_interval() -> None:
    with pytest.raises(ValueError, match="lower bound must not exceed upper bound"):
        MonteCarloResult(
            price=10.0,
            standard_error=0.1,
            confidence_interval_95=(10.2, 9.8),
            terminal_prices=np.array([100.0], dtype=np.float64),
        )


def test_monte_carlo_result_rejects_non_vector_terminal_prices() -> None:
    with pytest.raises(ValueError, match="terminal_prices must be a one-dimensional array"):
        MonteCarloResult(
            price=10.0,
            standard_error=0.1,
            confidence_interval_95=(9.8, 10.2),
            terminal_prices=np.array([[100.0]], dtype=np.float64),
        )


def test_monte_carlo_result_rejects_non_matrix_paths() -> None:
    with pytest.raises(ValueError, match="paths must be a two-dimensional array"):
        MonteCarloResult(
            price=10.0,
            standard_error=0.1,
            confidence_interval_95=(9.8, 10.2),
            terminal_prices=np.array([100.0], dtype=np.float64),
            paths=np.array([100.0], dtype=np.float64),
        )
