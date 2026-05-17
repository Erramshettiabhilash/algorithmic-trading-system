"""Alternative-data feature alignment for market models."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alternative_data.macro import MacroFeatureConfig, create_macro_features
from alternative_data.sentiment import aggregate_text_sentiment
from alternative_data.trends import normalize_search_interest


@dataclass(frozen=True)
class AlternativeFeatureConfig:
    """Configuration for aligning alternative data to market timestamps."""

    sentiment_frequency: str = "1D"
    search_window: int = 20
    macro_config: MacroFeatureConfig = field(default_factory=MacroFeatureConfig)
    fill_method: str = "ffill"


class AlternativeFeatureBuilder:
    """Build model-ready alternative-data features aligned to a market index."""

    def __init__(self, config: AlternativeFeatureConfig | None = None) -> None:
        self.config = config or AlternativeFeatureConfig()

    def build(
        self,
        market_index: pd.DatetimeIndex,
        *,
        sentiment_events: pd.DataFrame | None = None,
        search_interest: pd.DataFrame | None = None,
        macro_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Create one aligned alternative-data feature matrix."""

        if len(market_index) == 0:
            raise ValueError("market_index cannot be empty")

        blocks: list[pd.DataFrame] = []
        if sentiment_events is not None:
            blocks.append(
                aggregate_text_sentiment(
                    sentiment_events,
                    frequency=self.config.sentiment_frequency,
                )
            )
        if search_interest is not None:
            blocks.append(
                normalize_search_interest(search_interest, window=self.config.search_window)
            )
        if macro_data is not None:
            blocks.append(create_macro_features(macro_data, self.config.macro_config))

        if not blocks:
            raise ValueError("at least one alternative-data source is required")

        combined = pd.concat(blocks, axis=1, sort=True).sort_index()
        aligned = combined.reindex(market_index)
        if self.config.fill_method == "ffill":
            aligned = aligned.ffill()
        elif self.config.fill_method != "none":
            raise ValueError("fill_method must be 'ffill' or 'none'")

        return aligned.fillna(0.0)
