"""Input validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def require_positive(value: float, name: str) -> None:
    """Raise ValueError if a value is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive. Received {value}.")


def require_non_negative(value: float, name: str) -> None:
    """Raise ValueError if a value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Received {value}.")


def require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], name: str) -> None:
    """Raise ValueError when a DataFrame is missing required columns."""

    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def require_datetime_index(frame: pd.DataFrame | pd.Series, name: str) -> None:
    """Raise TypeError when an object does not use a DatetimeIndex."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a pandas DatetimeIndex")


def require_aligned_indexes(
    first: pd.DataFrame | pd.Series,
    second: pd.DataFrame | pd.Series,
    first_name: str,
    second_name: str,
) -> None:
    """Raise ValueError when two pandas objects do not share identical indexes."""

    if not first.index.equals(second.index):
        raise ValueError(f"{first_name} and {second_name} must have identical indexes")


def require_finite_values(frame: pd.DataFrame | pd.Series, name: str) -> None:
    """Raise ValueError when a pandas object contains NaN or infinite values."""

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
