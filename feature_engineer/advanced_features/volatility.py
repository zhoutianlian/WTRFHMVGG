"""
Advanced Volatility Features

This module implements sophisticated volatility calculations including
adaptive volatility and Kaufman's Adaptive Moving Average (KAMA).
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional
from .base import BaseAdvancedFeature


class AdaptiveVolatilityFeature(BaseAdvancedFeature):
    """
    Adaptive volatility that adjusts based on market conditions
    
    This feature calculates volatility that adapts to changing market regimes
    by combining short-term and long-term volatilities based on their ratio.
    """
    
    def __init__(self, 
                 price_col: str = 'price',
                 short_window: int = 6,
                 long_window: int = 24,
                 result_col: str = 'vol_adaptive'):
        super().__init__('adaptive_volatility')
        self.price_col = price_col
        self.short_window = short_window
        self.long_window = long_window
        self.result_col = result_col
        self.required_columns = [price_col]
        
        # Store parameters
        self.parameters = {
            'price_col': price_col,
            'short_window': short_window,
            'long_window': long_window,
            'result_col': result_col
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive volatility"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        
        # Calculate returns
        returns = df[self.price_col].pct_change()
        
        # Short and long term volatilities
        short_vol = returns.rolling(
            window=self.short_window, 
            min_periods=1
        ).std()
        
        long_vol = returns.rolling(
            window=self.long_window, 
            min_periods=1
        ).std()
        
        # Volatility ratio (measure of regime change)
        vol_ratio = short_vol / (long_vol + 1e-10)
        
        # Adaptive weight - higher weight to short-term vol during volatile periods
        weight = np.clip(vol_ratio, 0.3, 1.0)
        
        # Weighted average of volatilities
        adaptive_vol = weight * short_vol + (1 - weight) * long_vol
        
        # Annualize volatility (assuming hourly data)
        df[self.result_col] = adaptive_vol * np.sqrt(24 * 365)
        
        # Add supporting columns
        df[f'{self.result_col}_short'] = short_vol * np.sqrt(24 * 365)
        df[f'{self.result_col}_long'] = long_vol * np.sqrt(24 * 365)
        df[f'{self.result_col}_ratio'] = vol_ratio
        df[f'{self.result_col}_weight'] = weight
        
        result_columns = [
            self.result_col,
            f'{self.result_col}_short',
            f'{self.result_col}_long', 
            f'{self.result_col}_ratio',
            f'{self.result_col}_weight'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class KAMAFeature(BaseAdvancedFeature):
    """
    Kaufman's Adaptive Moving Average
    
    KAMA adapts to price action by using an efficiency ratio to modify
    the smoothing constant between fast and slow periods.
    """
    
    def __init__(self,
                 value_col: str,
                 window: int = 10,
                 fast_ema: int = 2,
                 slow_ema: int = 30,
                 result_col: str = None):
        super().__init__('kama')
        self.value_col = value_col
        self.window = window
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.result_col = result_col or f'{value_col}_kama'
        self.required_columns = [value_col]
        
        # Store parameters
        self.parameters = {
            'value_col': value_col,
            'window': window,
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'result_col': self.result_col
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate KAMA"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        series = df[self.value_col]
        
        # Calculate direction (net change over window)
        direction = (series - series.shift(self.window)).abs()
        
        # Calculate volatility (sum of absolute changes)
        volatility = series.diff().abs().rolling(window=self.window).sum()
        
        # Efficiency Ratio (trend strength)
        er = direction / (volatility + 1e-10)
        
        # Smoothing Constants
        fast_sc = 2 / (self.fast_ema + 1)
        slow_sc = 2 / (self.slow_ema + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        kama = pd.Series(index=series.index, dtype=float)
        kama.iloc[0] = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
        
        # Calculate KAMA iteratively
        for i in range(1, len(series)):
            if pd.notna(sc.iloc[i]) and pd.notna(series.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
            else:
                kama.iloc[i] = kama.iloc[i-1]
        
        # Add results to dataframe
        df[self.result_col] = kama
        df[f'{self.result_col}_efficiency'] = er
        df[f'{self.result_col}_smoothing'] = sc
        
        # Calculate KAMA-based signals
        df[f'{self.result_col}_signal'] = np.where(
            series > kama, 1,  # Above KAMA = bullish
            np.where(series < kama, -1, 0)  # Below KAMA = bearish
        )
        
        # KAMA slope (trend direction)
        kama_slope = kama.diff()
        df[f'{self.result_col}_slope'] = kama_slope
        df[f'{self.result_col}_trend'] = np.where(
            kama_slope > 0, 1,  # Rising KAMA = uptrend
            np.where(kama_slope < 0, -1, 0)  # Falling KAMA = downtrend
        )
        
        result_columns = [
            self.result_col,
            f'{self.result_col}_efficiency',
            f'{self.result_col}_smoothing',
            f'{self.result_col}_signal',
            f'{self.result_col}_slope',
            f'{self.result_col}_trend'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class GARCHVolatilityFeature(BaseAdvancedFeature):
    """
    GARCH-like volatility estimation
    
    Simplified GARCH(1,1) model for volatility estimation
    that accounts for volatility clustering.
    """
    
    def __init__(self,
                 price_col: str = 'price',
                 alpha: float = 0.1,
                 beta: float = 0.85,
                 omega: float = 0.05,
                 result_col: str = 'vol_garch'):
        super().__init__('garch_volatility')
        self.price_col = price_col
        self.alpha = alpha  # Weight for previous squared return
        self.beta = beta   # Weight for previous variance
        self.omega = omega # Long-term variance
        self.result_col = result_col
        self.required_columns = [price_col]
        
        # Store parameters
        self.parameters = {
            'price_col': price_col,
            'alpha': alpha,
            'beta': beta,
            'omega': omega,
            'result_col': result_col
        }
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GARCH-like volatility"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
            
        df = self.preprocess(df.copy())
        
        # Calculate returns
        returns = df[self.price_col].pct_change()
        
        # Initialize variance series
        variance = pd.Series(index=returns.index, dtype=float)
        
        # Initial variance (sample variance of first 30 observations)
        initial_var = returns.iloc[:30].var() if len(returns) >= 30 else returns.var()
        variance.iloc[0] = initial_var
        
        # GARCH(1,1) variance calculation
        for i in range(1, len(returns)):
            if pd.notna(returns.iloc[i-1]):
                variance.iloc[i] = (
                    self.omega + 
                    self.alpha * (returns.iloc[i-1] ** 2) + 
                    self.beta * variance.iloc[i-1]
                )
            else:
                variance.iloc[i] = variance.iloc[i-1]
        
        # Volatility is square root of variance
        volatility = np.sqrt(variance)
        
        # Annualize volatility
        df[self.result_col] = volatility * np.sqrt(24 * 365)
        df[f'{self.result_col}_variance'] = variance * (24 * 365)
        
        # Calculate volatility persistence
        persistence = self.alpha + self.beta
        df[f'{self.result_col}_persistence'] = persistence
        
        result_columns = [
            self.result_col,
            f'{self.result_col}_variance',
            f'{self.result_col}_persistence'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df