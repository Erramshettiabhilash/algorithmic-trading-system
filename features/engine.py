"""Orchestrates the complete quantitative feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessing import clean_ohlcv
from features.base import FeatureConfig, validate_feature_frame
from features.market_structure import add_market_structure_features
from features.momentum import add_momentum_features
from features.returns import add_return_features
from features.volatility import add_volatility_features
from features.volume import add_volume_features


class QuantFeatureEngine:
    """Build a production-style feature matrix from clean OHLCV bars."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create the full feature matrix for one asset."""

        clean = clean_ohlcv(prices)
        config = self.config
        feature_blocks = [
            add_return_features(clean, config.return_windows),
            add_momentum_features(
                clean,
                momentum_windows=config.momentum_windows,
                rsi_window=config.rsi_window,
                ema_fast=config.ema_fast,
                ema_slow=config.ema_slow,
                macd_signal=config.macd_signal,
            ),
            add_volatility_features(
                clean,
                volatility_windows=config.volatility_windows,
                atr_window=config.atr_window,
            ),
            add_volume_features(clean, window=config.volume_window),
            add_market_structure_features(
                clean,
                structure_window=config.structure_window,
                fractal_order=config.fractal_order,
            ),
        ]
        features = pd.concat(feature_blocks, axis=1).replace([np.inf, -np.inf], np.nan)

        if config.drop_na:
            features = features.dropna()

        validate_feature_frame(features)
        return features


def build_multi_asset_feature_panel(
    assets: dict[str, pd.DataFrame],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build a MultiIndex-column feature panel for several assets."""

    if not assets:
        raise ValueError("at least one asset frame is required")

    engine = QuantFeatureEngine(config)
    feature_frames = {symbol: engine.transform(frame) for symbol, frame in assets.items()}
    return pd.concat(feature_frames, axis=1, join="inner").dropna().sort_index()
