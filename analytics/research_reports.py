"""Research report generation for predictive trading analytics."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def generate_model_evaluation_report(
    output_path: str | Path,
    *,
    title: str,
    metrics: Any,
    notes: list[str] | None = None,
) -> Path:
    """Write a Markdown model evaluation report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_table = _metrics_to_frame(metrics)
    body = [f"# {title}", "", "## Metrics", "", _to_markdown_table(metric_table)]
    if notes:
        body.extend(["", "## Notes", ""])
        body.extend([f"- {note}" for note in notes])
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def generate_factor_analytics_report(
    output_path: str | Path,
    *,
    exposure: pd.Series,
    risk_contribution: pd.Series,
    title: str = "Factor Analytics Report",
) -> Path:
    """Write a Markdown factor exposure and risk contribution report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exposure_table = exposure.rename("exposure").to_frame()
    risk_table = risk_contribution.rename("risk_contribution").to_frame()
    body = [
        f"# {title}",
        "",
        "## Portfolio Factor Exposure",
        "",
        _to_markdown_table(exposure_table),
        "",
        "## Risk Contribution",
        "",
        _to_markdown_table(risk_table),
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _metrics_to_frame(metrics: Any) -> pd.DataFrame:
    """Convert dataclass, mapping, or Series metrics into a display table."""

    if is_dataclass(metrics):
        values = asdict(metrics)
    elif isinstance(metrics, pd.Series):
        values = metrics.to_dict()
    elif isinstance(metrics, dict):
        values = metrics
    else:
        values = dict(metrics)
    return pd.Series(values, name="value").to_frame()


def _to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional dependencies."""

    display = frame.reset_index().rename(columns={"index": "metric"})
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
