"""Feature engineering interfaces for quantitative research."""

from features.base import FeatureConfig, FeatureEngineer, validate_feature_frame
from features.engine import QuantFeatureEngine, build_multi_asset_feature_panel
from features.market_structure import add_market_structure_features
from features.momentum import add_momentum_features, relative_strength_index
from features.returns import add_return_features
from features.volatility import add_volatility_features, average_true_range
from features.volume import add_volume_features, on_balance_volume

__all__ = [
    "FeatureConfig",
    "FeatureEngineer",
    "QuantFeatureEngine",
    "add_market_structure_features",
    "add_momentum_features",
    "add_return_features",
    "add_volatility_features",
    "add_volume_features",
    "average_true_range",
    "build_multi_asset_feature_panel",
    "on_balance_volume",
    "relative_strength_index",
    "validate_feature_frame",
]
