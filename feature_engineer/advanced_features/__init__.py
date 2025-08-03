"""
Advanced Feature Engineering Module

This module extends the existing BTC feature engineering pipeline with
sophisticated statistical and technical analysis features.

Features included:
- Adaptive Volatility
- KAMA (Kaufman's Adaptive Moving Average)
- ROC Dynamics (Velocity/Acceleration)
- Momentum Indicators (RSI, Stochastic)
- Spike Ratio Analysis
- Kalman Filter-based Slope Estimation
- Advanced Distribution Analysis
"""

from .base import BaseAdvancedFeature
from .volatility import AdaptiveVolatilityFeature, KAMAFeature
from .momentum import ROCDynamicsFeature, MomentumIndicatorsFeature
from .statistical import SpikeRatioFeature, KalmanSlopeFeature
from .registry import ADVANCED_FEATURES_REGISTRY

__version__ = "1.0.0"
__all__ = [
    "BaseAdvancedFeature",
    "AdaptiveVolatilityFeature", 
    "KAMAFeature",
    "ROCDynamicsFeature",
    "MomentumIndicatorsFeature", 
    "SpikeRatioFeature",
    "KalmanSlopeFeature",
    "ADVANCED_FEATURES_REGISTRY"
]