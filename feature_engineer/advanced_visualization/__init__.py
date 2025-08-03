"""
Advanced Visualization Module

Professional visualization components for advanced feature analysis
with dark theme styling and interactive capabilities.
"""

from .config import COLORS, CHART_STYLE
from .dashboard import FinancialDashboard
from .interactive import DistributionAnalyzer, CorrelationAnalyzer
from .charts import AdvancedCharts

__version__ = "1.0.0"
__all__ = [
    "COLORS",
    "CHART_STYLE", 
    "FinancialDashboard",
    "DistributionAnalyzer",
    "CorrelationAnalyzer",
    "AdvancedCharts"
]