"""Alternative data research modules."""

from alternative_data.features import AlternativeFeatureBuilder, AlternativeFeatureConfig
from alternative_data.macro import MacroFeatureConfig, create_macro_features
from alternative_data.sentiment import (
    FinBERTSentimentAnalyzer,
    TextSentimentRecord,
    VaderSentimentAnalyzer,
    aggregate_text_sentiment,
)
from alternative_data.trends import GoogleTrendsClient, normalize_search_interest

__all__ = [
    "AlternativeFeatureBuilder",
    "AlternativeFeatureConfig",
    "FinBERTSentimentAnalyzer",
    "GoogleTrendsClient",
    "MacroFeatureConfig",
    "TextSentimentRecord",
    "VaderSentimentAnalyzer",
    "aggregate_text_sentiment",
    "create_macro_features",
    "normalize_search_interest",
]
