# fhmv/data_processing/__init__.py

"""
FHMV Data Processing Package.

This package contains modules for data ingestion and initial feature calculation
as defined in the FHMV project.
"""

from .data_ingestion_preprocessing_module import DataIngestionPreprocessingModule

__all__ = [
    "DataIngestionPreprocessingModule"
]