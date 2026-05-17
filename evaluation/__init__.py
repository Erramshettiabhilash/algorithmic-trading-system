"""Model and strategy evaluation metrics."""

from evaluation.ensemble_report import (
    EnsembleComparison,
    compare_ensemble_to_base_models,
    ensemble_information_coefficient,
    prediction_correlation_matrix,
)
from evaluation.metrics import (
    information_coefficient,
    information_ratio,
    max_drawdown,
    rolling_information_coefficient,
    sharpe_ratio,
)
from evaluation.model_report import FactorModelEvaluation, evaluate_factor_predictions
from evaluation.targets import (
    TargetSpec,
    build_supervised_dataset,
    create_direction_target,
    create_forward_return_target,
)
from evaluation.validation import (
    TemporalSplit,
    ValidationWindow,
    expanding_window_indices,
    expanding_window_splits,
    rolling_window_indices,
    rolling_window_splits,
    temporal_train_test_split,
)
from evaluation.walk_forward import (
    WalkForwardConfig,
    WalkForwardFoldResult,
    WalkForwardResult,
    analyze_alpha_decay,
    analyze_model_stability,
    analyze_prediction_drift,
    run_walk_forward_research,
    summarize_walk_forward_folds,
)

__all__ = [
    "EnsembleComparison",
    "FactorModelEvaluation",
    "TargetSpec",
    "TemporalSplit",
    "ValidationWindow",
    "WalkForwardConfig",
    "WalkForwardFoldResult",
    "WalkForwardResult",
    "analyze_alpha_decay",
    "analyze_model_stability",
    "analyze_prediction_drift",
    "build_supervised_dataset",
    "compare_ensemble_to_base_models",
    "create_direction_target",
    "create_forward_return_target",
    "ensemble_information_coefficient",
    "evaluate_factor_predictions",
    "expanding_window_indices",
    "expanding_window_splits",
    "information_coefficient",
    "information_ratio",
    "max_drawdown",
    "prediction_correlation_matrix",
    "rolling_information_coefficient",
    "rolling_window_indices",
    "rolling_window_splits",
    "run_walk_forward_research",
    "sharpe_ratio",
    "summarize_walk_forward_folds",
    "temporal_train_test_split",
]
