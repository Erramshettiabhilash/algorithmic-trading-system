"""Tests for Step 2 stochastic-process foundations."""

from __future__ import annotations

import numpy as np
import pytest

from simulations.stochastic_processes import (
    brownian_increments,
    brownian_motion_paths,
    geometric_brownian_motion_paths,
    log_returns,
    time_grid,
)


def test_time_grid_includes_start_and_end() -> None:
    grid = time_grid(maturity=1.0, steps=4)

    np.testing.assert_allclose(grid, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_brownian_increments_shape() -> None:
    increments = brownian_increments(maturity=1.0, steps=252, n_paths=10, seed=7)

    assert increments.shape == (10, 252)


def test_brownian_paths_start_at_zero() -> None:
    paths = brownian_motion_paths(maturity=1.0, steps=12, n_paths=5, seed=7)

    assert paths.shape == (5, 13)
    np.testing.assert_allclose(paths[:, 0], np.zeros(5))


def test_gbm_paths_start_at_spot_and_stay_positive() -> None:
    paths = geometric_brownian_motion_paths(
        spot=100.0,
        drift=0.05,
        volatility=0.2,
        maturity=1.0,
        steps=252,
        n_paths=100,
        seed=7,
    )

    np.testing.assert_allclose(paths[:, 0], np.full(100, 100.0))
    assert np.all(paths > 0.0)


def test_zero_volatility_gbm_is_deterministic() -> None:
    paths = geometric_brownian_motion_paths(
        spot=100.0,
        drift=0.05,
        volatility=0.0,
        maturity=1.0,
        steps=4,
        n_paths=3,
        seed=7,
    )
    expected = 100.0 * np.exp(0.05 * time_grid(maturity=1.0, steps=4))

    np.testing.assert_allclose(paths, np.tile(expected, (3, 1)))


def test_log_returns() -> None:
    prices = np.array([100.0, 105.0, 110.0])
    returns = log_returns(prices)

    np.testing.assert_allclose(returns, np.log(prices[1:] / prices[:-1]))


def test_log_returns_reject_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        log_returns(np.array([100.0, 0.0, 101.0]))

