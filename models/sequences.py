"""Sequence construction utilities for deep time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    """Windowed time-series tensors with prediction timestamps."""

    x: np.ndarray
    y: np.ndarray
    index: pd.DatetimeIndex
    feature_names: list[str]
    target_name: str


def create_sequence_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    sequence_length: int,
) -> SequenceDataset:
    """Create sliding-window sequences from aligned features and target.

    Each sample uses feature rows ``t - sequence_length + 1`` through ``t`` to predict
    the target aligned at timestamp ``t``. That mirrors a real forecasting setup:
    the model sees only historical observations up to the decision time.
    """

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")

    aligned = features.join(target.rename(target.name or "target"), how="inner").dropna()
    if len(aligned) < sequence_length:
        raise ValueError("not enough rows to create one sequence")

    target_name = target.name or "target"
    x_frame = aligned.drop(columns=[target_name])
    y_series = aligned[target_name]

    x_values = x_frame.to_numpy(dtype=np.float32)
    y_values = y_series.to_numpy(dtype=np.float32)
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    timestamps: list[pd.Timestamp] = []

    for end_position in range(sequence_length - 1, len(aligned)):
        start_position = end_position - sequence_length + 1
        sequences.append(x_values[start_position : end_position + 1])
        targets.append(float(y_values[end_position]))
        timestamps.append(aligned.index[end_position])

    return SequenceDataset(
        x=np.stack(sequences).astype(np.float32),
        y=np.asarray(targets, dtype=np.float32),
        index=pd.DatetimeIndex(timestamps),
        feature_names=list(x_frame.columns),
        target_name=target_name,
    )
