"""Google Trends alternative-data utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GoogleTrendsConfig:
    """Configuration for Google Trends requests."""

    timeframe: str = "today 3-m"
    geo: str = ""
    timezone_offset_minutes: int = 0


class GoogleTrendsClient:
    """Small pytrends wrapper for search-interest data."""

    def __init__(self, config: GoogleTrendsConfig | None = None) -> None:
        self.config = config or GoogleTrendsConfig()

    def fetch_interest(self, keywords: list[str]) -> pd.DataFrame:
        """Fetch Google Trends interest-over-time data for keywords."""

        if not keywords:
            raise ValueError("at least one keyword is required")

        try:
            from pytrends.request import TrendReq
        except ImportError as exc:
            raise ImportError("Install pytrends to fetch Google Trends data.") from exc

        trend_req = TrendReq(tz=self.config.timezone_offset_minutes)
        trend_req.build_payload(
            keywords,
            timeframe=self.config.timeframe,
            geo=self.config.geo,
        )
        data = trend_req.interest_over_time()
        if "isPartial" in data.columns:
            data = data.drop(columns=["isPartial"])
        return data.sort_index()


def normalize_search_interest(search_interest: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Create search-interest normalized level and change features."""

    if search_interest.empty:
        raise ValueError("search_interest cannot be empty")

    data = search_interest.sort_index().astype(float)
    output = pd.DataFrame(index=data.index)
    for column in data.columns:
        rolling_mean = data[column].rolling(window, min_periods=max(3, window // 4)).mean()
        rolling_std = data[column].rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
        safe_std = rolling_std.replace(0.0, pd.NA)
        output[f"{column}_search_zscore"] = (data[column] - rolling_mean) / safe_std
        output[f"{column}_search_change"] = data[column].pct_change().replace([pd.NA], 0.0)

    return output.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
