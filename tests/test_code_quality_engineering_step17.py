"""Step 17 code-quality and architecture checks for the AI quant platform."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from utils import (
    require_aligned_indexes,
    require_columns,
    require_datetime_index,
    require_finite_values,
)

AI_PUBLIC_PACKAGES = [
    "alternative_data",
    "data",
    "evaluation",
    "explainability",
    "features",
    "models",
    "optimization",
    "regime",
    "risk",
    "rl",
    "trading",
    "visualization",
]

REPRESENTATIVE_MODULES = [
    "alternative_data.features",
    "alternative_data.macro",
    "alternative_data.sentiment",
    "alternative_data.trends",
    "data.preprocessing",
    "data.sources",
    "evaluation.metrics",
    "evaluation.walk_forward",
    "explainability.shap_engine",
    "features.engine",
    "models.ensemble",
    "models.lstm",
    "models.xgboost_factor",
    "optimization.optuna_engine",
    "regime.detection",
    "risk.factor_model",
    "rl.environment",
    "trading.backtest",
    "trading.metrics",
    "trading.portfolio",
    "trading.signals",
    "visualization.research_charts",
]


@pytest.mark.parametrize("package_name", AI_PUBLIC_PACKAGES)
def test_ai_public_package_exports_exist(package_name: str) -> None:
    package = importlib.import_module(package_name)

    assert hasattr(package, "__all__")
    for exported_name in package.__all__:
        assert hasattr(package, exported_name), f"{package_name}.{exported_name} is missing"


@pytest.mark.parametrize("package_name", AI_PUBLIC_PACKAGES)
def test_ai_public_packages_include_typing_marker(package_name: str) -> None:
    package = importlib.import_module(package_name)
    package_path = Path(package.__file__).parent

    assert (package_path / "py.typed").exists(), f"{package_name} is missing py.typed"


@pytest.mark.parametrize("module_name", REPRESENTATIVE_MODULES)
def test_representative_ai_modules_have_docstrings(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert inspect.getdoc(module), f"{module_name} is missing a module docstring"


@pytest.mark.parametrize(
    "module_name",
    [
        "alternative_data.macro",
        "data.preprocessing",
        "evaluation.metrics",
        "risk.factor_model",
        "trading.signals",
        "trading.metrics",
    ],
)
def test_representative_public_functions_have_type_hints(module_name: str) -> None:
    module = importlib.import_module(module_name)
    functions = [
        function
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and function.__module__ == module_name
    ]

    assert functions
    for function in functions:
        _assert_callable_has_annotations(function)


def test_shared_validation_helpers_cover_common_numerical_guards() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
    aligned = pd.Series([0.1, 0.2, 0.3], index=index)

    require_columns(frame, ["close"], "frame")
    require_datetime_index(frame, "frame")
    require_aligned_indexes(frame, aligned, "frame", "aligned")
    require_finite_values(frame, "frame")

    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(frame, ["open"], "frame")
    with pytest.raises(TypeError, match="DatetimeIndex"):
        require_datetime_index(pd.Series([1.0, 2.0]), "series")
    with pytest.raises(ValueError, match="identical indexes"):
        require_aligned_indexes(frame, aligned.iloc[1:], "frame", "aligned")
    with pytest.raises(ValueError, match="finite"):
        require_finite_values(pd.Series([1.0, float("nan")]), "series")


def _assert_callable_has_annotations(function: Callable[..., object]) -> None:
    signature = inspect.signature(function)

    for parameter in signature.parameters.values():
        assert parameter.annotation is not inspect.Signature.empty, (
            f"{function.__name__}.{parameter.name} is missing a type annotation"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"{function.__name__} is missing a return annotation"
    )
