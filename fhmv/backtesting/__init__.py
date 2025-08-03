# fhmv/backtesting/__init__.py

"""
FHMV Backtesting Package.

This package contains the BacktestingModule for simulating FHMV trading
strategies, accounting for costs, and calculating comprehensive
performance metrics.
"""

from .backtesting_module import BacktestingModule

__all__ = [
    "BacktestingModule"
]
