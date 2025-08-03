# fhmv/__init__.py

"""
FHMV (Factor-based Hidden Markov Model Variant) Project Package.

This package encompasses the entire FHMV system, including data processing,
feature engineering, the core FHMV engine, signal generation, risk management,
backtesting, utilities, and configuration management.
"""

# Expose key modules or classes at the top level of the fhmv package if desired
# For example:
# from .core_engine.fhmv_core_engine import FHMVCoreEngine
# from .config_loader import ConfigLoader

# Or just ensure sub-packages are importable:
# import fhmv.data_processing
# import fhmv.features
# import fhmv.core_engine
# import fhmv.signals
# import fhmv.risk_management
# import fhmv.backtesting
# import fhmv.utils
# import fhmv.config_loader # This would not work directly if config_loader is a file.

# To make ConfigLoader directly available as `from fhmv import ConfigLoader`:
from .config_loader import ConfigLoader

__all__ = [
    "ConfigLoader",
    # Add other top-level exposed classes/modules here if needed
]