"""Model-card metadata for explainable financial machine learning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCard:
    """Human-readable summary of a model's purpose, horizon, and risks."""

    name: str
    target: str
    horizon: str
    intended_use: str
    main_risks: tuple[str, ...]
