"""Tests for Step 4 targets and temporal research design."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import (
    build_supervised_dataset,
    create_direction_target,
    create_forward_return_target,
    expanding_window_indices,
    expanding_window_splits,
    information_ratio,
    rolling_information_coefficient,
    rolling_window_indices,
    rolling_window_splits,
    temporal_train_test_split,
)


def _prices(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    close = 100.0 + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000 + np.arange(rows),
        },
        index=dates,
    )


def _features(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "feature_a": np.arange(rows, dtype=float),
            "feature_b": np.sin(np.arange(rows, dtype=float)),
        },
        index=dates,
    )


def test_forward_return_target_is_aligned_to_decision_timestamp() -> None:
    prices = _prices()
    target = create_forward_return_target(prices, horizon=2, log_return=False)

    expected = prices["close"].iloc[2] / prices["close"].iloc[0] - 1.0

    assert target.iloc[0] == expected
    assert target.iloc[-1] != target.iloc[-1]
    assert target.name == "forward_return_2"


def test_direction_target_creates_binary_future_direction() -> None:
    target = create_direction_target(_prices(), horizon=1)

    assert set(target.dropna().unique()) == {1.0}
    assert target.iloc[-1] != target.iloc[-1]


def test_build_supervised_dataset_drops_unavailable_future_target() -> None:
    features = _features()
    target = create_forward_return_target(_prices(), horizon=3)

    x, y = build_supervised_dataset(features, target)

    assert len(x) == len(features) - 3
    assert x.index.equals(y.index)


def test_temporal_train_test_split_preserves_time_order() -> None:
    features = _features()
    target = create_forward_return_target(_prices(), horizon=1)

    split = temporal_train_test_split(features, target, test_size=5)

    assert len(split.x_test) == 5
    assert split.x_train.index.max() < split.x_test.index.min()
    assert split.y_train.index.max() < split.y_test.index.min()


def test_expanding_window_indices_grow_training_history() -> None:
    windows = list(expanding_window_indices(20, initial_train_size=8, test_size=4))

    assert windows[0].train_start == 0
    assert windows[0].train_end == 8
    assert windows[1].train_end == 12
    assert windows[1].test_start == 12


def test_rolling_window_indices_keep_fixed_training_length() -> None:
    windows = list(rolling_window_indices(20, train_size=8, test_size=4))

    assert windows[0].train_start == 0
    assert windows[1].train_start == 4
    assert windows[1].train_end - windows[1].train_start == 8


def test_expanding_and_rolling_splitters_return_aligned_data() -> None:
    features = _features()
    target = create_forward_return_target(_prices(), horizon=1)

    expanding = list(expanding_window_splits(features, target, initial_train_size=10, test_size=5))
    rolling = list(rolling_window_splits(features, target, train_size=10, test_size=5))

    assert expanding[0].x_train.index.equals(expanding[0].y_train.index)
    assert rolling[0].x_test.index.equals(rolling[0].y_test.index)


def test_rolling_ic_and_information_ratio_are_finite() -> None:
    predictions = pd.Series([0.1, 0.2, -0.1, 0.4, 0.3, -0.2])
    realized = pd.Series([0.0, 0.3, -0.2, 0.5, 0.2, -0.1])

    rolling_ic = rolling_information_coefficient(predictions, realized, window=3)
    ir = information_ratio(rolling_ic.dropna())

    assert rolling_ic.dropna().between(-1.0, 1.0).all()
    assert np.isfinite(ir)
