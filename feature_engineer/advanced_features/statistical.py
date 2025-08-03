"""
Advanced Statistical Features

This module implements sophisticated statistical analysis features including
spike ratio analysis and Kalman filter-based slope estimation.
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional
from .base import BaseAdvancedFeature


class SpikeRatioFeature(BaseAdvancedFeature):
    """
    Spike ratio using various moving average methods
    
    Calculates the ratio of current value to its moving average,
    providing insights into price spikes and deviations from trend.
    """
    
    def __init__(self,
                 value_col: str,
                 window: int = 24,
                 method: Literal['sma', 'ema', 'kama'] = 'kama',
                 spike_ratio_col: str = None):
        super().__init__('spike_ratio')
        self.value_col = value_col
        self.window = window
        self.method = method
        self.spike_ratio_col = spike_ratio_col or f'{value_col}_spike_{method}'
        self.required_columns = [value_col]
        
        # Store parameters
        self.parameters = {
            'value_col': value_col,
            'window': window,
            'method': method,
            'spike_ratio_col': self.spike_ratio_col
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spike ratio"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        values = df[self.value_col]
        
        # Calculate moving average based on method
        if self.method == 'sma':
            ma = values.rolling(window=self.window).mean()
        elif self.method == 'ema':
            ma = values.ewm(span=self.window, adjust=False).mean()
        elif self.method == 'kama':
            # Use simplified KAMA calculation
            ma = self._calculate_kama(values, self.window)
        else:
            ma = values.rolling(window=self.window).mean()
        
        # Calculate spike ratio
        spike_ratio = values / (ma + 1e-10)
        df[self.spike_ratio_col] = spike_ratio
        
        # Calculate additional spike metrics
        ma_col = f'{self.spike_ratio_col}_ma'
        df[ma_col] = ma
        
        # Spike strength (deviation from 1.0)
        spike_strength = (spike_ratio - 1.0).abs()
        df[f'{self.spike_ratio_col}_strength'] = spike_strength
        
        # Spike direction
        spike_direction = np.where(
            spike_ratio > 1.05, 1,  # Upward spike (5% above MA)
            np.where(spike_ratio < 0.95, -1, 0)  # Downward spike (5% below MA)
        )
        df[f'{self.spike_ratio_col}_direction'] = spike_direction
        
        # Rolling spike statistics
        df[f'{self.spike_ratio_col}_rolling_mean'] = spike_ratio.rolling(window=24).mean()
        df[f'{self.spike_ratio_col}_rolling_std'] = spike_ratio.rolling(window=24).std()
        
        # Z-score of spike ratio
        rolling_mean = spike_ratio.rolling(window=24).mean()
        rolling_std = spike_ratio.rolling(window=24).std()
        spike_zscore = (spike_ratio - rolling_mean) / (rolling_std + 1e-10)
        df[f'{self.spike_ratio_col}_zscore'] = spike_zscore
        
        # Extreme spike detection
        extreme_spikes = (spike_zscore.abs() > 2.0).astype(int)
        df[f'{self.spike_ratio_col}_extreme'] = extreme_spikes
        
        result_columns = [
            self.spike_ratio_col,
            ma_col,
            f'{self.spike_ratio_col}_strength',
            f'{self.spike_ratio_col}_direction',
            f'{self.spike_ratio_col}_rolling_mean',
            f'{self.spike_ratio_col}_rolling_std',
            f'{self.spike_ratio_col}_zscore',
            f'{self.spike_ratio_col}_extreme'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df
    
    def _calculate_kama(self, series: pd.Series, window: int) -> pd.Series:
        """Simplified KAMA calculation for spike ratio"""
        direction = (series - series.shift(window)).abs()
        volatility = series.diff().abs().rolling(window=window).sum()
        er = direction / (volatility + 1e-10)
        
        # Smoothing constants
        fast_sc = 2 / 3
        slow_sc = 2 / 31
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        kama = pd.Series(index=series.index, dtype=float)
        kama.iloc[0] = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
        
        for i in range(1, len(series)):
            if pd.notna(sc.iloc[i]) and pd.notna(series.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
            else:
                kama.iloc[i] = kama.iloc[i-1]
                
        return kama


class KalmanSlopeFeature(BaseAdvancedFeature):
    """
    Kalman filter-based slope estimation
    
    Uses a Kalman filter to estimate the underlying trend (slope) of a time series,
    providing a smooth estimate of the rate of change.
    """
    
    def __init__(self,
                 price_col: str = 'price',
                 process_noise: float = 0.01,
                 measurement_noise: float = 1.0,
                 result_col: str = 'kalman_slope'):
        super().__init__('kalman_slope')
        self.price_col = price_col
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.result_col = result_col
        self.required_columns = [price_col]
        
        # Store parameters
        self.parameters = {
            'price_col': price_col,
            'process_noise': process_noise,
            'measurement_noise': measurement_noise,
            'result_col': result_col
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Kalman filter slope"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        
        # Initialize Kalman filter state
        prices = df[self.price_col].dropna()
        if len(prices) == 0:
            # Handle empty data case
            df[self.result_col] = 0
            df[f'{self.result_col}_filtered_price'] = df[self.price_col]
            df[f'{self.result_col}_innovation'] = 0
            return self.postprocess(df)
        
        # State: [position, velocity]
        x = np.array([[prices.iloc[0]], [0]])
        P = np.array([[1000, 0], [0, 1000]])  # Initial covariance
        
        # State transition matrix (constant velocity model)
        F = np.array([[1, 1], [0, 1]])
        
        # Measurement matrix (observe position only)
        H = np.array([[1, 0]])
        
        # Process noise covariance
        Q = np.array([[self.process_noise/4, self.process_noise/2],
                      [self.process_noise/2, self.process_noise]])
        
        # Measurement noise covariance
        R = np.array([[self.measurement_noise]])
        
        slopes = []
        filtered_prices = []
        innovations = []
        
        for i, price in enumerate(prices):
            # Prediction step
            x = F @ x
            P = F @ P @ F.T + Q
            
            # Update step
            y = price - H @ x  # Innovation
            S = H @ P @ H.T + R  # Innovation covariance
            K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P
            
            slopes.append(x[1, 0])  # Velocity component (slope)
            filtered_prices.append(x[0, 0])  # Position component (filtered price)
            innovations.append(y[0, 0])  # Innovation (prediction error)
        
        # Create result series with original index
        slope_series = pd.Series(index=df.index, dtype=float)
        filtered_price_series = pd.Series(index=df.index, dtype=float)
        innovation_series = pd.Series(index=df.index, dtype=float)
        
        # Fill in the calculated values
        valid_idx = df[self.price_col].notna()
        slope_series[valid_idx] = slopes
        filtered_price_series[valid_idx] = filtered_prices
        innovation_series[valid_idx] = innovations
        
        # Add results to dataframe
        df[self.result_col] = slope_series
        df[f'{self.result_col}_filtered_price'] = filtered_price_series
        df[f'{self.result_col}_innovation'] = innovation_series
        
        # Calculate additional slope-based features
        slope_smooth = slope_series.rolling(window=5).mean()
        df[f'{self.result_col}_smooth'] = slope_smooth
        
        # Slope direction signals
        slope_direction = np.where(
            slope_series > 0.01, 1,  # Strong positive slope
            np.where(slope_series < -0.01, -1, 0)  # Strong negative slope
        )
        df[f'{self.result_col}_direction'] = slope_direction
        
        # Slope acceleration (second derivative)
        slope_acceleration = slope_series.diff()
        df[f'{self.result_col}_acceleration'] = slope_acceleration
        
        # Slope volatility
        slope_volatility = slope_series.rolling(window=24).std()
        df[f'{self.result_col}_volatility'] = slope_volatility
        
        result_columns = [
            self.result_col,
            f'{self.result_col}_filtered_price',
            f'{self.result_col}_innovation',
            f'{self.result_col}_smooth',
            f'{self.result_col}_direction',
            f'{self.result_col}_acceleration',
            f'{self.result_col}_volatility'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class StatisticalMomentsFeature(BaseAdvancedFeature):
    """
    Rolling statistical moments (skewness, kurtosis, etc.)
    
    Calculates higher-order moments of price distributions over rolling windows
    to capture distribution shape characteristics.
    """
    
    def __init__(self,
                 value_col: str,
                 window: int = 24,
                 include_skewness: bool = True,
                 include_kurtosis: bool = True,
                 include_jarque_bera: bool = True):
        super().__init__('statistical_moments')
        self.value_col = value_col
        self.window = window
        self.include_skewness = include_skewness
        self.include_kurtosis = include_kurtosis
        self.include_jarque_bera = include_jarque_bera
        self.required_columns = [value_col]
        
        # Store parameters
        self.parameters = {
            'value_col': value_col,
            'window': window,
            'include_skewness': include_skewness,
            'include_kurtosis': include_kurtosis,
            'include_jarque_bera': include_jarque_bera
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical moments"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        values = df[self.value_col]
        returns = values.pct_change()
        
        result_columns = []
        
        if self.include_skewness:
            # Rolling skewness
            rolling_skew = returns.rolling(window=self.window).skew()
            df[f'{self.value_col}_skewness'] = rolling_skew
            result_columns.append(f'{self.value_col}_skewness')
            
            # Skewness signals
            skew_signals = np.where(
                rolling_skew > 0.5, 1,    # Positive skew (right tail)
                np.where(rolling_skew < -0.5, -1, 0)  # Negative skew (left tail)
            )
            df[f'{self.value_col}_skew_signal'] = skew_signals
            result_columns.append(f'{self.value_col}_skew_signal')
        
        if self.include_kurtosis:
            # Rolling kurtosis
            rolling_kurt = returns.rolling(window=self.window).kurt()
            df[f'{self.value_col}_kurtosis'] = rolling_kurt
            result_columns.append(f'{self.value_col}_kurtosis')
            
            # Excess kurtosis (kurtosis - 3)
            excess_kurt = rolling_kurt - 3
            df[f'{self.value_col}_excess_kurtosis'] = excess_kurt
            result_columns.append(f'{self.value_col}_excess_kurtosis')
            
            # High kurtosis periods (fat tails)
            high_kurt = (excess_kurt > 1).astype(int)
            df[f'{self.value_col}_fat_tails'] = high_kurt
            result_columns.append(f'{self.value_col}_fat_tails')
        
        if self.include_jarque_bera:
            # Simplified Jarque-Bera normality test statistic
            # JB = n/6 * (S^2 + (K-3)^2/4) where S=skewness, K=kurtosis
            if self.include_skewness and self.include_kurtosis:
                n = self.window
                jb_stat = (n/6) * (rolling_skew**2 + (excess_kurt**2)/4)
                df[f'{self.value_col}_jarque_bera'] = jb_stat
                result_columns.append(f'{self.value_col}_jarque_bera')
                
                # Non-normal periods (high JB statistic)
                non_normal = (jb_stat > 5.99).astype(int)  # 5% critical value
                df[f'{self.value_col}_non_normal'] = non_normal
                result_columns.append(f'{self.value_col}_non_normal')
        
        # Rolling coefficient of variation
        rolling_mean = returns.rolling(window=self.window).mean()
        rolling_std = returns.rolling(window=self.window).std()
        cv = rolling_std / (rolling_mean.abs() + 1e-10)
        df[f'{self.value_col}_cv'] = cv
        result_columns.append(f'{self.value_col}_cv')
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df