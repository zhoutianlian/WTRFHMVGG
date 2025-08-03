# fhmv/signals/__init__.py

"""
FHMV Signal Generation Package.

This package contains modules for generating trading signals based on
FHMV states and feature analysis, including CLAD logic for RC regimes
and L3 signal assessment for RHA/VT regimes.
"""

from .clad_signal_generator import CLADSignalGenerator
from .l3_signal_assessor import L3SignalAssessor
from .signal_generation_module import SignalGenerationModule

__all__ = [
    "CLADSignalGenerator",
    "L3SignalAssessor",
    "SignalGenerationModule"
]