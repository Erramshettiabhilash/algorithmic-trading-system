"""Temporal validation utilities for financial machine learning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Container for one chronological train/test split."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ValidationWindow:
    """Integer-location boundaries for a walk-forward validation window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def split(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> TemporalSplit:
        """Slice aligned features and target for this validation window."""

        return TemporalSplit(
            x_train=features.iloc[self.train_start : self.train_end],
            x_test=features.iloc[self.test_start : self.test_end],
            y_train=target.iloc[self.train_start : self.train_end],
            y_test=target.iloc[self.test_start : self.test_end],
        )


def temporal_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float | int = 0.2,
    min_train_size: int = 1,
) -> TemporalSplit:
    """Split data chronologically without shuffling."""

    x_aligned, y_aligned = _align_features_target(features, target)
    n_rows = len(x_aligned)
    test_rows = _resolve_test_size(test_size, n_rows)
    train_rows = n_rows - test_rows

    if train_rows < min_train_size:
        raise ValueError("not enough training rows for requested temporal split")

    return TemporalSplit(
        x_train=x_aligned.iloc[:train_rows],
        x_test=x_aligned.iloc[train_rows:],
        y_train=y_aligned.iloc[:train_rows],
        y_test=y_aligned.iloc[train_rows:],
    )


def expanding_window_splits(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[TemporalSplit]:
    """Yield expanding-window validation splits.

    The training set starts at the beginning and grows over time. This is useful when
    older observations remain relevant and the model should learn from all history.
    """

    x_aligned, y_aligned = _align_features_target(features, target)
    for window in expanding_window_indices(
        len(x_aligned),
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
    ):
        yield window.split(x_aligned, y_aligned)


def rolling_window_splits(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[TemporalSplit]:
    """Yield rolling-window validation splits.

    The training window keeps a fixed length and moves forward. This is useful when old
    data may be stale because the market regime has changed.
    """

    x_aligned, y_aligned = _align_features_target(features, target)
    for window in rolling_window_indices(
        len(x_aligned),
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    ):
        yield window.split(x_aligned, y_aligned)


def expanding_window_indices(
    n_rows: int,
    *,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[ValidationWindow]:
    """Yield integer boundaries for expanding-window validation."""

    _validate_window_inputs(n_rows, initial_train_size, test_size)
    step = step_size or test_size
    train_end = initial_train_size

    while train_end + test_size <= n_rows:
        yield ValidationWindow(
            train_start=0,
            train_end=train_end,
            test_start=train_end,
            test_end=train_end + test_size,
        )
        train_end += step


def rolling_window_indices(
    n_rows: int,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> Iterator[ValidationWindow]:
    """Yield integer boundaries for rolling-window validation."""

    _validate_window_inputs(n_rows, train_size, test_size)
    step = step_size or test_size
    train_start = 0

    while train_start + train_size + test_size <= n_rows:
        train_end = train_start + train_size
        yield ValidationWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=train_end + test_size,
        )
        train_start += step


def _align_features_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and target by timestamp and remove missing rows."""

    aligned = features.join(target.rename(target.name or "target"), how="inner").dropna()
    if aligned.empty:
        raise ValueError("no aligned feature/target rows available")

    target_name = target.name or "target"
    return aligned.drop(columns=[target_name]), aligned[target_name]


def _resolve_test_size(test_size: float | int, n_rows: int) -> int:
    """Convert fractional or absolute test size into a row count."""

    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("fractional test_size must be between 0 and 1")
        rows = int(round(n_rows * test_size))
    else:
        rows = int(test_size)

    if rows < 1 or rows >= n_rows:
        raise ValueError("test_size must leave at least one train and one test row")
    return rows


def _validate_window_inputs(n_rows: int, train_size: int, test_size: int) -> None:
    """Validate walk-forward window sizes."""

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if train_size < 1:
        raise ValueError("train size must be positive")
    if test_size < 1:
        raise ValueError("test_size must be positive")
    if train_size + test_size > n_rows:
        raise ValueError("not enough rows for requested train/test window")
