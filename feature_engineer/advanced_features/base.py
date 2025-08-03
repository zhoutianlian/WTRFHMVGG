"""
Base class for advanced feature engineering

This module provides the foundation for advanced feature calculations
that extend the existing BTC feature engineering pipeline.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime


class BaseAdvancedFeature(ABC):
    """Abstract base class for advanced feature engineering"""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.logger = logging.getLogger(f"{__name__}.{feature_name}")
        self.required_columns = []
        self.parameters = {}
        self.calculation_metadata = {}
        
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the advanced feature"""
        pass
    
    @abstractmethod
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        pass
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data before calculation"""
        # Handle infinities and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Ensure time column is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Postprocess data after calculation"""
        # Fill NaN values with appropriate method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill then backward fill for time series data
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Final fallback to zero for any remaining NaNs
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def get_feature_stats(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Get comprehensive statistics for the calculated feature"""
        if column not in df.columns:
            return {'error': f'Column {column} not found in dataframe'}
        
        series = df[column].dropna()
        
        if len(series) == 0:
            return {'error': f'No valid data in column {column}'}
        
        try:
            stats = {
                'count': len(series),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'range': float(series.max() - series.min()),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                'cv': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                'non_zero_count': int((series != 0).sum()),
                'zero_count': int((series == 0).sum())
            }
            
            # Add percentiles
            percentiles = [1, 5, 10, 90, 95, 99]
            for p in percentiles:
                stats[f'p{p}'] = float(series.quantile(p/100))
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating stats for {column}: {e}")
            return {'error': str(e)}
    
    def validate_dependencies(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that required dependencies (existing features) are present
        
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = []
        
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        return len(missing_columns) == 0, missing_columns
    
    def get_calculation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the feature calculation"""
        return {
            'feature_name': self.feature_name,
            'parameters': self.parameters,
            'required_columns': self.required_columns,
            'calculation_time': datetime.now().isoformat(),
            'metadata': self.calculation_metadata
        }
    
    def log_calculation_start(self):
        """Log the start of feature calculation"""
        self.logger.info(f"Starting calculation of {self.feature_name}")
        self.logger.debug(f"Parameters: {self.parameters}")
        self.logger.debug(f"Required columns: {self.required_columns}")
    
    def log_calculation_end(self, df: pd.DataFrame, result_columns: List[str]):
        """Log the end of feature calculation"""
        self.logger.info(f"Completed calculation of {self.feature_name}")
        self.logger.info(f"Processed {len(df)} rows")
        self.logger.info(f"Generated columns: {result_columns}")
        
        # Log basic stats for each generated column
        for col in result_columns:
            if col in df.columns:
                stats = self.get_feature_stats(df, col)
                if 'error' not in stats:
                    self.logger.debug(f"{col} stats: mean={stats['mean']:.4f}, "
                                    f"std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")


class AdvancedFeatureCalculator:
    """Helper class for advanced feature calculations"""
    
    @staticmethod
    def calculate_smoothed_derivative(series: pd.Series, 
                                    order: int = 1,
                                    method: str = 'savgol',
                                    window: int = 5) -> pd.Series:
        """
        Calculate smoothed derivative of a time series
        
        Args:
            series: Input time series
            order: Derivative order (1 for velocity, 2 for acceleration)
            method: Smoothing method ('savgol', 'gaussian', 'ewm')
            window: Window size for smoothing
        """
        from scipy import signal
        
        valid_idx = ~series.isna()
        valid_data = series[valid_idx].values
        
        if len(valid_data) < window:
            return pd.Series(index=series.index, dtype=float)
        
        try:
            if method == 'savgol':
                poly_order = max(order + 1, 3)
                window_size = window if window % 2 == 1 else window + 1
                
                smoothed = signal.savgol_filter(
                    valid_data,
                    window_length=min(window_size, len(valid_data)),
                    polyorder=min(poly_order, min(window_size, len(valid_data)) - 1),
                    deriv=order
                )
                
            elif method == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                sigma = window / 4
                smoothed_data = gaussian_filter1d(valid_data, sigma=sigma)
                smoothed = np.gradient(smoothed_data)
                if order == 2:
                    smoothed = np.gradient(smoothed)
                    
            elif method == 'ewm':
                temp_series = pd.Series(valid_data)
                smoothed_series = temp_series.ewm(span=window, adjust=False).mean()
                smoothed = smoothed_series.diff().values
                if order == 2:
                    smoothed = pd.Series(smoothed).diff().values
                    
            else:
                # Fallback to simple gradient
                smoothed = np.gradient(valid_data)
                if order == 2:
                    smoothed = np.gradient(smoothed)
                    
        except Exception:
            # Fallback to simple gradient on error
            smoothed = np.gradient(valid_data)
            if order == 2:
                smoothed = np.gradient(smoothed)
        
        result = pd.Series(index=series.index, dtype=float)
        result[valid_idx] = smoothed
        return result
    
    @staticmethod
    def normalize_series(series: pd.Series, method: str = 'z_score') -> pd.Series:
        """
        Normalize a time series using various methods
        
        Args:
            series: Input series to normalize
            method: Normalization method ('z_score', 'min_max', 'robust')
        """
        if method == 'z_score':
            return (series - series.mean()) / (series.std() + 1e-10)
        elif method == 'min_max':
            return (series - series.min()) / (series.max() - series.min() + 1e-10)
        elif method == 'robust':
            median = series.median()
            mad = (series - median).abs().median()
            return (series - median) / (mad + 1e-10)
        else:
            return series
    
    @staticmethod
    def rolling_correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Calculate rolling correlation between two series"""
        return x.rolling(window=window).corr(y)
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a time series
        
        Returns:
            Boolean series indicating outliers (True = outlier)
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'z_score':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        
        elif method == 'modified_z_score':
            median = series.median()
            mad = (series - median).abs().median()
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            return pd.Series(False, index=series.index)