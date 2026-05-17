"""News and social-media sentiment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class TextSentimentRecord:
    """Timestamped sentiment score for a news or social text item."""

    timestamp: pd.Timestamp
    source: str
    text: str
    sentiment: float


class SentimentAnalyzer(Protocol):
    """Protocol implemented by text sentiment models."""

    def score(self, text: str) -> float:
        """Return a normalized sentiment score where positive values are bullish."""


class VaderSentimentAnalyzer:
    """Lightweight sentiment analyzer for headlines, Reddit posts, and tweets."""

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError(
                "Install vaderSentiment for lightweight NLP sentiment: "
                "python -m pip install vaderSentiment"
            ) from exc

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        """Return VADER compound sentiment in [-1, 1]."""

        return float(self._analyzer.polarity_scores(text)["compound"])


class FinBERTSentimentAnalyzer:
    """Lazy FinBERT sentiment wrapper for finance-specific news sentiment.

    The class loads model weights only when constructed. Unit tests avoid this path so
    the repository remains fast and offline-testable.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Install transformers to use FinBERT sentiment.") from exc

        self._pipeline: Any = pipeline("sentiment-analysis", model=model_name)

    def score(self, text: str) -> float:
        """Return FinBERT sentiment mapped to [-1, 1]."""

        result = self._pipeline(text, truncation=True)[0]
        label = str(result["label"]).lower()
        confidence = float(result["score"])
        if "positive" in label:
            return confidence
        if "negative" in label:
            return -confidence
        return 0.0


def score_text_items(
    texts: pd.DataFrame,
    analyzer: SentimentAnalyzer,
    *,
    timestamp_column: str = "timestamp",
    text_column: str = "text",
    source_column: str = "source",
) -> pd.DataFrame:
    """Score timestamped text items and return a sentiment event table."""

    required = {timestamp_column, text_column}
    missing = required.difference(texts.columns)
    if missing:
        raise ValueError(f"missing required text columns: {sorted(missing)}")

    output = texts.copy()
    output[timestamp_column] = pd.to_datetime(output[timestamp_column])
    if source_column not in output.columns:
        output[source_column] = "unknown"

    output["sentiment"] = output[text_column].astype(str).map(analyzer.score)
    return output[[timestamp_column, source_column, text_column, "sentiment"]]


def aggregate_text_sentiment(
    sentiment_events: pd.DataFrame,
    *,
    frequency: str = "1D",
    timestamp_column: str = "timestamp",
    sentiment_column: str = "sentiment",
) -> pd.DataFrame:
    """Aggregate event-level sentiment into time-series features."""

    if timestamp_column not in sentiment_events.columns:
        raise ValueError(f"timestamp column '{timestamp_column}' not found")
    if sentiment_column not in sentiment_events.columns:
        raise ValueError(f"sentiment column '{sentiment_column}' not found")

    events = sentiment_events.copy()
    events[timestamp_column] = pd.to_datetime(events[timestamp_column])
    events = events.set_index(timestamp_column).sort_index()
    grouped = events[sentiment_column].resample(frequency)

    features = pd.DataFrame(
        {
            "sentiment_mean": grouped.mean(),
            "sentiment_std": grouped.std(ddof=0),
            "sentiment_count": grouped.count(),
        }
    )
    features["sentiment_zscore_20"] = (
        (features["sentiment_mean"] - features["sentiment_mean"].rolling(20).mean())
        / features["sentiment_mean"].rolling(20).std(ddof=0)
    )
    return features.fillna(0.0)
