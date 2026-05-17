"""Tests for the Step 7 volatility-surface analytics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.volatility_surface import VolatilitySurfaceAnalyzer
from visualization.charts import plot_volatility_surface


def _surface_data() -> pd.DataFrame:
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [0.25, 0.5, 1.0]
    rows = []

    for maturity in maturities:
        for strike in strikes:
            moneyness = strike / 100.0
            smile_component = 0.06 * (moneyness - 1.0) ** 2
            skew_component = -0.08 * (moneyness - 1.0)
            term_component = 0.02 * maturity
            implied_volatility = 0.20 + smile_component + skew_component + term_component
            rows.append(
                {
                    "strike": strike,
                    "maturity": maturity,
                    "implied_volatility": implied_volatility,
                }
            )

    return pd.DataFrame(rows)


def test_analyzer_computes_moneyness_from_spot() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)

    assert "moneyness" in analyzer.data.columns
    assert analyzer.data.loc[analyzer.data["strike"] == 100.0, "moneyness"].iloc[0] == 1.0


def test_available_maturities_and_strikes_are_sorted() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)

    np.testing.assert_allclose(analyzer.available_maturities(), np.array([0.25, 0.5, 1.0]))
    np.testing.assert_allclose(
        analyzer.available_strikes(),
        np.array([80.0, 90.0, 100.0, 110.0, 120.0]),
    )


def test_smile_returns_one_sorted_maturity_slice() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)
    smile = analyzer.smile(maturity=0.5)

    assert list(smile["maturity"].unique()) == [0.5]
    assert list(smile["strike"]) == [80.0, 90.0, 100.0, 110.0, 120.0]


def test_term_structure_by_strike_returns_sorted_maturities() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)
    term = analyzer.term_structure_by_strike(strike=100.0)

    assert list(term["strike"].unique()) == [100.0]
    assert list(term["maturity"]) == [0.25, 0.5, 1.0]


def test_atm_term_structure_selects_closest_to_atm() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)
    atm = analyzer.atm_term_structure()

    assert list(atm["strike"]) == [100.0, 100.0, 100.0]
    assert list(atm["maturity"]) == [0.25, 0.5, 1.0]


def test_skew_estimate_is_negative_for_equity_style_surface() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)
    skew = analyzer.skew(maturity=1.0)

    assert skew.maturity == 1.0
    assert skew.points == 5
    assert skew.slope < 0.0


def test_surface_grid_shape() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)
    grid = analyzer.surface_grid()

    assert grid.implied_volatilities.shape == (3, 5)
    np.testing.assert_allclose(grid.maturities, np.array([0.25, 0.5, 1.0]))
    np.testing.assert_allclose(grid.strikes, np.array([80.0, 90.0, 100.0, 110.0, 120.0]))


def test_surface_grid_rejects_incomplete_rectangular_grid() -> None:
    incomplete = _surface_data().iloc[:-1]
    analyzer = VolatilitySurfaceAnalyzer(data=incomplete, spot=100.0)

    with pytest.raises(ValueError, match="incomplete"):
        analyzer.surface_grid()


def test_analyzer_requires_spot_when_moneyness_absent() -> None:
    with pytest.raises(ValueError, match="spot is required"):
        VolatilitySurfaceAnalyzer(data=_surface_data())


def test_plot_volatility_surface_returns_plotly_figure() -> None:
    analyzer = VolatilitySurfaceAnalyzer(data=_surface_data(), spot=100.0)
    fig = plot_volatility_surface(analyzer.surface_grid())

    assert len(fig.data) == 1
    assert fig.data[0].type == "surface"

