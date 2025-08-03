"""
Unified Dashboard System

Combines visualization functions from:
- dashboard/ (existing Dash apps and Flask backend)
- feature_engineer/advanced_visualization/ (professional visualizations)
- feature_engineer/advanced_features/ (statistical analysis)

Into one cohesive website with separate pages for each feature category.
"""

from .main_dashboard import UnifiedDashboard
from .feature_pages import (
    RPNFeaturePage, BinningFeaturePage, DominanceFeaturePage, 
    SignalFeaturePage, AdvancedFeaturePage, StatisticalAnalysisPage
)
from .components import NavigationBar, FeatureSelector, TimeRangeSelector
from .visualization_engine import VisualizationEngine

__version__ = "1.0.0"

__all__ = [
    "UnifiedDashboard",
    "RPNFeaturePage", 
    "BinningFeaturePage",
    "DominanceFeaturePage",
    "SignalFeaturePage", 
    "AdvancedFeaturePage",
    "StatisticalAnalysisPage",
    "NavigationBar",
    "FeatureSelector", 
    "TimeRangeSelector",
    "VisualizationEngine"
]