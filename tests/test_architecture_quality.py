"""Architecture and public API quality checks."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable

import pytest

PUBLIC_PACKAGES = ["analytics", "models", "simulations", "utils", "visualization"]


@pytest.mark.parametrize("package_name", PUBLIC_PACKAGES)
def test_public_package_all_exports_exist(package_name: str) -> None:
    package = importlib.import_module(package_name)

    assert hasattr(package, "__all__")
    for exported_name in package.__all__:
        assert hasattr(package, exported_name), f"{package_name}.{exported_name} is not exported"


@pytest.mark.parametrize(
    "module_name",
    [
        "analytics.greeks",
        "analytics.implied_volatility",
        "analytics.market_making",
        "analytics.reporting",
        "analytics.stress_testing",
        "analytics.volatility_surface",
        "models.black_scholes",
        "models.sabr",
        "simulations.hedging",
        "simulations.monte_carlo",
        "simulations.stochastic_processes",
        "utils.instruments",
        "utils.market",
        "utils.validation",
        "visualization.charts",
    ],
)
def test_public_module_has_docstring(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert inspect.getdoc(module)


@pytest.mark.parametrize(
    "module_name",
    [
        "analytics.reporting",
        "models.black_scholes",
        "simulations.stochastic_processes",
        "utils.validation",
    ],
)
def test_public_functions_have_type_hints(module_name: str) -> None:
    module = importlib.import_module(module_name)
    functions = [
        obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module_name
    ]

    assert functions
    for function in functions:
        _assert_callable_has_annotations(function)


def _assert_callable_has_annotations(function: Callable[..., object]) -> None:
    signature = inspect.signature(function)

    for parameter in signature.parameters.values():
        assert parameter.annotation is not inspect.Signature.empty, (
            f"{function.__name__}.{parameter.name} is missing a type annotation"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"{function.__name__} is missing a return annotation"
    )
