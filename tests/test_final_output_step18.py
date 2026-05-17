"""Final output checks for Step 18 deliverables."""

from __future__ import annotations

import json
from pathlib import Path


def test_final_documentation_exists() -> None:
    required_docs = [
        Path("README.md"),
        Path("docs/AI_QUANT_FINAL_PROJECT_SUMMARY.md"),
        Path("docs/FINAL_PROJECT_SUMMARY.md"),
        Path("docs/AI_QUANT_ARCHITECTURE.md"),
        Path("docs/AI_QUANT_CODE_QUALITY.md"),
    ]

    for document in required_docs:
        assert document.exists()
        assert document.stat().st_size > 0


def test_ai_quant_demo_artifacts_exist_after_final_demo_run() -> None:
    output_dir = Path("results/examples/ai_quant_demo")
    required_artifacts = [
        "synthetic_ohlcv.csv",
        "ai_quant_demo_summary.json",
        "walk_forward_summary.csv",
        "shap_importance.csv",
        "prediction_vs_actual.png",
        "equity_curve.png",
        "drawdown_curve.png",
        "performance_dashboard.png",
        "shap_importance.png",
        "regime_classification.png",
        "portfolio_allocation.html",
        "model_evaluation_report.md",
        "factor_analytics_report.md",
    ]

    for artifact in required_artifacts:
        path = output_dir / artifact
        assert path.exists()
        assert path.stat().st_size > 0


def test_ai_quant_demo_summary_contains_core_sections() -> None:
    summary_path = Path("results/examples/ai_quant_demo/ai_quant_demo_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "model_evaluation" in summary
    assert "trading_performance" in summary
    assert "walk_forward_evaluation" in summary
    assert "top_shap_features" in summary
    assert "portfolio_factor_exposure" in summary
