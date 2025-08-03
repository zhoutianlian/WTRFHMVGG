# fhmv/core_engine/__init__.py

"""
FHMV Core Engine Package.

This package contains the FHMVCoreEngine class, which implements the
training (EM algorithm) and inference (Viterbi, state probabilities)
for the Factor-based Hidden Markov Model with Student's t-Mixture Model emissions.
"""

from .fhmv_core_engine import FHMVCoreEngine

__all__ = [
    "FHMVCoreEngine"
]