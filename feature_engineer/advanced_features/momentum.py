"""
Advanced Momentum Features

This module implements sophisticated momentum-based features including
rate of change dynamics and technical momentum indicators.
"""

import pandas as pd
import numpy as np
from scipy import signal
from typing import Literal, Optional
from .base import BaseAdvancedFeature, AdvancedFeatureCalculator


class ROCDynamicsFeature(BaseAdvancedFeature):
    """
    Rate of Change dynamics (velocity and acceleration)
    
    Calculates smoothed derivatives to capture velocity and acceleration
    of price movements, providing insights into momentum dynamics.
    """
    
    def __init__(self,
                 value_col: str,
                 velocity_col: str = None,
                 acceleration_col: str = None,
                 smoothing_method: Literal['savgol', 'gaussian', 'ewm'] = 'savgol',
                 window: int = 5,
                 include_normalized: bool = True):
        super().__init__('roc_dynamics')
        self.value_col = value_col
        self.velocity_col = velocity_col or f'{value_col}_velocity'
        self.acceleration_col = acceleration_col or f'{value_col}_acceleration'
        self.smoothing_method = smoothing_method
        self.window = window
        self.include_normalized = include_normalized
        self.required_columns = [value_col]
        
        # Store parameters
        self.parameters = {
            'value_col': value_col,
            'velocity_col': self.velocity_col,
            'acceleration_col': self.acceleration_col,
            'smoothing_method': smoothing_method,
            'window': window,
            'include_normalized': include_normalized
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ROC dynamics"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        values = df[self.value_col]
        
        # Calculate smoothed derivatives
        velocity = AdvancedFeatureCalculator.calculate_smoothed_derivative(
            values, order=1, method=self.smoothing_method, window=self.window
        )
        acceleration = AdvancedFeatureCalculator.calculate_smoothed_derivative(
            values, order=2, method=self.smoothing_method, window=self.window
        )
        
        # Add to dataframe
        df[self.velocity_col] = velocity
        df[self.acceleration_col] = acceleration
        
        result_columns = [self.velocity_col, self.acceleration_col]
        
        # Add normalized versions if requested
        if self.include_normalized:
            value_magnitude = values.abs() + 1e-10
            df[f'{self.velocity_col}_normalized'] = velocity / value_magnitude
            
            velocity_magnitude = velocity.abs() + 1e-10
            df[f'{self.acceleration_col}_normalized'] = acceleration / velocity_magnitude
            
            result_columns.extend([
                f'{self.velocity_col}_normalized',
                f'{self.acceleration_col}_normalized'
            ])
        
        # Add momentum signals
        df[f'{self.velocity_col}_signal'] = np.where(
            velocity > 0, 1,  # Positive velocity = bullish momentum
            np.where(velocity < 0, -1, 0)  # Negative velocity = bearish momentum
        )
        
        df[f'{self.acceleration_col}_signal'] = np.where(
            acceleration > 0, 1,  # Positive acceleration = momentum increasing
            np.where(acceleration < 0, -1, 0)  # Negative acceleration = momentum decreasing
        )
        
        # Combined momentum score
        momentum_score = np.sign(velocity) + np.sign(acceleration)
        df[f'{self.value_col}_momentum_score'] = momentum_score
        
        result_columns.extend([
            f'{self.velocity_col}_signal',
            f'{self.acceleration_col}_signal',
            f'{self.value_col}_momentum_score'
        ])
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class MomentumIndicatorsFeature(BaseAdvancedFeature):
    """
    Traditional momentum indicators: RSI and Stochastic Oscillator
    
    Implements classic technical analysis momentum indicators with
    additional signal generation and trend analysis.
    """
    
    def __init__(self,
                 value_col: str,
                 window: int = 14,
                 include_rsi: bool = True,
                 include_stoch: bool = True,
                 stoch_k_window: int = 14,
                 stoch_d_window: int = 3):
        super().__init__('momentum_indicators')
        self.value_col = value_col
        self.window = window
        self.include_rsi = include_rsi
        self.include_stoch = include_stoch
        self.stoch_k_window = stoch_k_window
        self.stoch_d_window = stoch_d_window
        self.required_columns = [value_col]
        
        # Store parameters
        self.parameters = {
            'value_col': value_col,
            'window': window,
            'include_rsi': include_rsi,
            'include_stoch': include_stoch,
            'stoch_k_window': stoch_k_window,
            'stoch_d_window': stoch_d_window
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        values = df[self.value_col]
        result_columns = []
        
        if self.include_rsi:
            # Calculate RSI
            delta = values.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
            
            # Use Wilder's smoothing for more stable RSI
            gain_ema = delta.where(delta > 0, 0).ewm(alpha=1/self.window, adjust=False).mean()
            loss_ema = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.window, adjust=False).mean()
            
            rs = gain_ema / (loss_ema + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            df[f'{self.value_col}_rsi'] = rsi
            result_columns.append(f'{self.value_col}_rsi')
            
            # RSI signals
            df[f'{self.value_col}_rsi_overbought'] = (rsi > 70).astype(int)
            df[f'{self.value_col}_rsi_oversold'] = (rsi < 30).astype(int)
            df[f'{self.value_col}_rsi_signal'] = np.where(
                rsi > 70, -1,  # Overbought = sell signal
                np.where(rsi < 30, 1, 0)  # Oversold = buy signal
            )
            
            # RSI divergence (simplified)
            rsi_slope = rsi.diff()
            price_slope = values.diff()
            rsi_divergence = np.where(
                (price_slope > 0) & (rsi_slope < 0), -1,  # Bearish divergence
                np.where((price_slope < 0) & (rsi_slope > 0), 1, 0)  # Bullish divergence
            )
            df[f'{self.value_col}_rsi_divergence'] = rsi_divergence
            
            result_columns.extend([
                f'{self.value_col}_rsi_overbought',
                f'{self.value_col}_rsi_oversold',
                f'{self.value_col}_rsi_signal',
                f'{self.value_col}_rsi_divergence'
            ])
        
        if self.include_stoch:
            # Calculate Stochastic Oscillator
            low_min = values.rolling(window=self.stoch_k_window).min()
            high_max = values.rolling(window=self.stoch_k_window).max()
            
            # %K line
            stoch_k = 100 * (values - low_min) / (high_max - low_min + 1e-10)
            
            # %D line (smoothed %K)
            stoch_d = stoch_k.rolling(window=self.stoch_d_window).mean()
            
            df[f'{self.value_col}_stoch_k'] = stoch_k
            df[f'{self.value_col}_stoch_d'] = stoch_d
            result_columns.extend([
                f'{self.value_col}_stoch_k',
                f'{self.value_col}_stoch_d'
            ])
            
            # Stochastic signals
            df[f'{self.value_col}_stoch_overbought'] = ((stoch_k > 80) | (stoch_d > 80)).astype(int)
            df[f'{self.value_col}_stoch_oversold'] = ((stoch_k < 20) | (stoch_d < 20)).astype(int)
            
            # %K and %D crossover signals
            stoch_crossover = np.where(
                (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)), 1,  # Bullish crossover
                np.where((stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)), -1, 0)  # Bearish crossover
            )
            df[f'{self.value_col}_stoch_crossover'] = stoch_crossover
            
            result_columns.extend([
                f'{self.value_col}_stoch_overbought',
                f'{self.value_col}_stoch_oversold',
                f'{self.value_col}_stoch_crossover'
            ])
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class MACDFeature(BaseAdvancedFeature):
    """
    Moving Average Convergence Divergence (MACD)
    
    Classic trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """
    
    def __init__(self,
                 value_col: str,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 result_col: str = None):
        super().__init__('macd')
        self.value_col = value_col
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.result_col = result_col or f'{value_col}_macd'
        self.required_columns = [value_col]
        
        # Store parameters
        self.parameters = {
            'value_col': value_col,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'result_col': self.result_col
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        values = df[self.value_col]
        
        # Calculate exponential moving averages
        ema_fast = values.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = values.ewm(span=self.slow_period, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # MACD histogram
        macd_histogram = macd_line - signal_line
        
        # Add to dataframe
        df[self.result_col] = macd_line
        df[f'{self.result_col}_signal'] = signal_line
        df[f'{self.result_col}_histogram'] = macd_histogram
        
        # MACD signals
        macd_bullish = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        macd_bearish = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
        
        df[f'{self.result_col}_bullish_cross'] = macd_bullish
        df[f'{self.result_col}_bearish_cross'] = macd_bearish
        
        # Zero line crossover
        zero_cross_bullish = ((macd_line > 0) & (macd_line.shift(1) <= 0)).astype(int)
        zero_cross_bearish = ((macd_line < 0) & (macd_line.shift(1) >= 0)).astype(int)
        
        df[f'{self.result_col}_zero_cross_bullish'] = zero_cross_bullish
        df[f'{self.result_col}_zero_cross_bearish'] = zero_cross_bearish
        
        result_columns = [
            self.result_col,
            f'{self.result_col}_signal',
            f'{self.result_col}_histogram',
            f'{self.result_col}_bullish_cross',
            f'{self.result_col}_bearish_cross',
            f'{self.result_col}_zero_cross_bullish',
            f'{self.result_col}_zero_cross_bearish'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df