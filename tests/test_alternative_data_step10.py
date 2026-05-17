"""Tests for Step 10 alternative data engine."""

from __future__ import annotations

import pandas as pd
import pytest

from alternative_data import (
    AlternativeFeatureBuilder,
    MacroFeatureConfig,
    VaderSentimentAnalyzer,
    aggregate_text_sentiment,
    create_macro_features,
    normalize_search_interest,
)
from alternative_data.sentiment import score_text_items


class _RuleBasedAnalyzer:
    def score(self, text: str) -> float:
        lowered = text.lower()
        if "surge" in lowered or "beats" in lowered:
            return 0.8
        if "risk" in lowered or "misses" in lowered:
            return -0.6
        return 0.0


def _sentiment_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 09:00",
                    "2024-01-01 10:00",
                    "2024-01-02 09:00",
                    "2024-01-03 09:00",
                ]
            ),
            "source": ["news", "reddit", "twitter", "news"],
            "text": [
                "Earnings beats expectations",
                "Macro risk worries traders",
                "Bitcoin surge continues",
                "Company misses guidance",
            ],
        }
    )


def test_rule_based_text_scoring_and_daily_aggregation() -> None:
    scored = score_text_items(_sentiment_events(), _RuleBasedAnalyzer())
    daily = aggregate_text_sentiment(scored)

    assert "sentiment" in scored.columns
    assert daily.loc[pd.Timestamp("2024-01-01"), "sentiment_count"] == 2
    assert daily["sentiment_mean"].iloc[0] == pytest.approx(0.1)


def test_vader_sentiment_analyzer_returns_normalized_score() -> None:
    analyzer = VaderSentimentAnalyzer()
    positive = analyzer.score("The company reported excellent growth and strong demand.")
    negative = analyzer.score("The company warned of losses and weak demand.")

    assert -1.0 <= positive <= 1.0
    assert -1.0 <= negative <= 1.0
    assert positive > negative


def test_search_interest_features_are_normalized() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    search = pd.DataFrame({"bitcoin": range(30), "gold": list(reversed(range(30)))}, index=index)

    features = normalize_search_interest(search, window=10)

    assert "bitcoin_search_zscore" in features.columns
    assert "gold_search_change" in features.columns
    assert features.isna().sum().sum() == 0


def test_macro_features_include_levels_changes_and_zscores() -> None:
    index = pd.date_range("2023-01-01", periods=18, freq="ME")
    macro = pd.DataFrame(
        {
            "cpi": [3.0 + i * 0.1 for i in range(18)],
            "pmi": [50.0 + (-1) ** i for i in range(18)],
            "interest_rate": [5.0, 5.0, 5.1, 5.2, 5.2, 5.3] * 3,
        },
        index=index,
    )

    features = create_macro_features(macro, MacroFeatureConfig(zscore_window=6))

    assert {"cpi_level", "pmi_change", "interest_rate_zscore"}.issubset(features.columns)
    assert features.isna().sum().sum() == 0


def test_alternative_feature_builder_aligns_sources_to_market_index() -> None:
    market_index = pd.date_range("2024-01-01", periods=10, freq="D")
    scored = score_text_items(_sentiment_events(), _RuleBasedAnalyzer())
    search = pd.DataFrame({"bitcoin": range(10)}, index=market_index)
    macro = pd.DataFrame(
        {"cpi": [3.1, 3.2]},
        index=pd.to_datetime(["2023-12-31", "2024-01-07"]),
    )

    features = AlternativeFeatureBuilder().build(
        market_index,
        sentiment_events=scored,
        search_interest=search,
        macro_data=macro,
    )

    assert features.index.equals(market_index)
    assert "sentiment_mean" in features.columns
    assert "bitcoin_search_zscore" in features.columns
    assert "cpi_level" in features.columns
    assert features.isna().sum().sum() == 0
