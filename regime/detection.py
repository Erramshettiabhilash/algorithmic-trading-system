"""Market regime detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeDetectionConfig:
    """Configuration for regime feature construction and unsupervised detectors."""

    return_window: int = 20
    volatility_window: int = 20
    trend_window: int = 20
    n_regimes: int = 3
    random_state: int = 42


def create_regime_feature_matrix(
    prices: pd.DataFrame,
    config: RegimeDetectionConfig | None = None,
) -> pd.DataFrame:
    """Create returns, volatility, and trend-strength features for regime detection."""

    cfg = config or RegimeDetectionConfig()
    if "close" not in prices.columns:
        raise ValueError("prices must contain a 'close' column")

    close = prices["close"].astype(float)
    log_return = np.log(close).diff()
    rolling_return = np.log(close / close.shift(cfg.return_window))
    rolling_volatility = log_return.rolling(cfg.volatility_window).std(ddof=1) * np.sqrt(252)
    trend_strength = rolling_return.abs() / rolling_volatility.replace(0.0, np.nan)

    matrix = pd.DataFrame(
        {
            "rolling_return": rolling_return,
            "rolling_volatility": rolling_volatility,
            "trend_strength": trend_strength,
        },
        index=prices.index,
    )
    return matrix.replace([np.inf, -np.inf], np.nan).dropna()


def classify_market_regimes(
    prices: pd.DataFrame,
    config: RegimeDetectionConfig | None = None,
) -> pd.Series:
    """Classify timestamps into ranging, trending, and high-volatility regimes."""

    features = create_regime_feature_matrix(prices, config)
    volatility_high = features["rolling_volatility"].rolling(60, min_periods=10).quantile(0.67)
    trend_high = features["trend_strength"].rolling(60, min_periods=10).quantile(0.67)

    regimes = pd.Series("ranging", index=features.index, name="market_regime")
    regimes = regimes.mask(features["trend_strength"] >= trend_high, "trending")
    regimes = regimes.mask(features["rolling_volatility"] >= volatility_high, "high_volatility")
    return regimes.ffill().fillna("ranging")


class ClusteringRegimeDetector:
    """KMeans-based unsupervised regime detector."""

    def __init__(self, config: RegimeDetectionConfig | None = None) -> None:
        self.config = config or RegimeDetectionConfig()
        self.scaler: Any | None = None
        self.model: Any | None = None
        self.regime_names: dict[int, str] = {}
        self.feature_names: list[str] = []

    def fit(self, regime_features: pd.DataFrame) -> ClusteringRegimeDetector:
        """Fit a KMeans detector and map clusters to finance-readable labels."""

        if regime_features.empty:
            raise ValueError("regime_features cannot be empty")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        self.feature_names = list(regime_features.columns)
        self.scaler = StandardScaler()
        values = self.scaler.fit_transform(regime_features)
        self.model = KMeans(
            n_clusters=self.config.n_regimes,
            n_init=10,
            random_state=self.config.random_state,
        )
        self.model.fit(values)
        labels = self.model.predict(values)
        self.regime_names = _name_clusters(regime_features, labels)
        return self

    def predict(self, regime_features: pd.DataFrame) -> pd.Series:
        """Predict regime labels for a feature matrix."""

        if self.model is None or self.scaler is None:
            raise ValueError("detector must be fit before predict")

        aligned = regime_features.loc[:, self.feature_names]
        labels = self.model.predict(self.scaler.transform(aligned))
        regimes = [self.regime_names.get(int(label), f"regime_{int(label)}") for label in labels]
        return pd.Series(regimes, index=aligned.index, name="cluster_regime")


class HMMRegimeDetector:
    """Gaussian Hidden Markov Model regime detector."""

    def __init__(self, config: RegimeDetectionConfig | None = None) -> None:
        self.config = config or RegimeDetectionConfig()
        self.scaler: Any | None = None
        self.model: Any | None = None
        self.regime_names: dict[int, str] = {}
        self.feature_names: list[str] = []

    def fit(self, regime_features: pd.DataFrame) -> HMMRegimeDetector:
        """Fit a Gaussian HMM to regime features."""

        if regime_features.empty:
            raise ValueError("regime_features cannot be empty")

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "Install hmmlearn to use HMMRegimeDetector: python -m pip install hmmlearn"
            ) from exc

        self.feature_names = list(regime_features.columns)
        self.scaler = StandardScaler()
        values = self.scaler.fit_transform(regime_features)
        self.model = GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.config.random_state,
        )
        self.model.fit(values)
        labels = self.model.predict(values)
        self.regime_names = _name_clusters(regime_features, labels)
        return self

    def predict(self, regime_features: pd.DataFrame) -> pd.Series:
        """Infer hidden regime states and map them to readable names."""

        if self.model is None or self.scaler is None:
            raise ValueError("detector must be fit before predict")

        aligned = regime_features.loc[:, self.feature_names]
        labels = self.model.predict(self.scaler.transform(aligned))
        regimes = [self.regime_names.get(int(label), f"regime_{int(label)}") for label in labels]
        return pd.Series(regimes, index=aligned.index, name="hmm_regime")


def _name_clusters(regime_features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Assign finance-readable names to unsupervised cluster labels."""

    diagnostics = regime_features.assign(cluster=labels).groupby("cluster").mean(numeric_only=True)
    names: dict[int, str] = {}

    if diagnostics.empty:
        return names

    high_vol_cluster = int(diagnostics["rolling_volatility"].idxmax())
    remaining_trends = diagnostics["trend_strength"].drop(index=high_vol_cluster, errors="ignore")
    trend_cluster = int(remaining_trends.idxmax())

    for label in diagnostics.index:
        int_label = int(label)
        if int_label == high_vol_cluster:
            names[int_label] = "high_volatility"
        elif int_label == trend_cluster:
            names[int_label] = "trending"
        else:
            names[int_label] = "ranging"

    return names
