"""
PostgreSQL-based BTC Feature Engineering Pipeline

A robust, scalable architecture for downloading BTC data from Glassnode,
processing features, and storing results in PostgreSQL database.

Main Components:
- DatabaseManager: PostgreSQL database operations with connection pooling
- BTCDataDownloader: Downloads data from Glassnode API with retry logic
- DataPreprocessor: Data preprocessing with outlier smoothing and normalization
- FeatureEngineer: Modular feature engineering pipeline using advanced_features registry
- BTCDataReader: Convenient interface for reading processed data
- BTCFeaturePipeline: Main orchestrator with CLI interface

Features:
- Incremental updates (only processes new data)
- Batch processing for large datasets
- Comprehensive error handling and logging
- Performance optimized with indexes and connection pooling
- Modular design - each component can be used independently
- Web integration ready with views and APIs

Usage:
    # Run full pipeline
    python main.py --full-pipeline
    
    # Initialize database
    python main.py --init-db
    
    # Download data only
    python main.py --download
    
    # Export training data
    python main.py --export --output training_data.csv
    
    # Get status
    python main.py --status

For detailed usage, see README.md
"""

from .db.database_manager import DatabaseManager
from .data_download.downloader import BTCDataDownloader
from .preprocessing.preprocessor import DataPreprocessor
from .advanced_features.pipeline import FeatureEngineer
from .advanced_features.statistical_analyzer import FeatureStatisticalAnalyzer
from .data_reader.reader import BTCDataReader
from .config.config_manager import ConfigManager
from .main import BTCFeaturePipeline

__version__ = "2.0.0"
__all__ = [
    "DatabaseManager",
    "BTCDataDownloader",
    "DataPreprocessor", 
    "FeatureEngineer",
    "FeatureStatisticalAnalyzer",
    "BTCDataReader",
    "ConfigManager",
    "BTCFeaturePipeline"
]