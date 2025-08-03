"""
Market Dominance Features

This module implements market dominance analysis features that determine
bull/bear/congestion regimes based on RPN and bin patterns.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from .base import BaseAdvancedFeature


class DominanceCalculationFeature(BaseAdvancedFeature):
    """
    Market Dominance Calculation Feature
    
    Calculates market dominance based on RPN values, bin indices,
    and liquidation differentials to determine market regime.
    """
    
    def __init__(self,
                 rpn_col: str = 'risk_priority_number',
                 bin_col: str = 'bin_index',
                 diff_col: str = 'diff_ls_cwt_kf',
                 result_col: str = 'dominance'):
        super().__init__('dominance_calculation')
        self.rpn_col = rpn_col
        self.bin_col = bin_col
        self.diff_col = diff_col
        self.result_col = result_col
        self.required_columns = [rpn_col, bin_col]
        
        # Store parameters
        self.parameters = {
            'rpn_col': rpn_col,
            'bin_col': bin_col,
            'diff_col': diff_col,
            'result_col': result_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market dominance"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        
        # Initialize dominance
        df[self.result_col] = 0
        
        # Basic dominance logic based on bin extremes
        if self.bin_col in df.columns:
            # Bull dominance at low bins (extreme liquidation dominance)
            df.loc[df[self.bin_col] <= 1, self.result_col] = 1
            # Bear dominance at high bins (extreme liquidation resistance)
            df.loc[df[self.bin_col] >= 7, self.result_col] = -1
        
        # Enhanced dominance logic with additional conditions
        if self.diff_col in df.columns:
            # Strengthen bull signals when FLL >> FSL
            bull_mask = (df[self.bin_col] <= 2) & (df[self.diff_col] > 0)
            df.loc[bull_mask, self.result_col] = 1
            
            # Strengthen bear signals when FSL >> FLL
            bear_mask = (df[self.bin_col] >= 6) & (df[self.diff_col] < 0)
            df.loc[bear_mask, self.result_col] = -1
        
        # Dominance strength calculation
        df['dominance_strength'] = np.abs(df[self.result_col])
        
        # Dominance confidence based on bin distance from center
        center_bin = 4  # Middle bin
        df['dominance_confidence'] = np.abs(df[self.bin_col] - center_bin) / center_bin
        
        # Combined dominance score
        df['dominance_score'] = df[self.result_col] * df['dominance_confidence']
        
        # Dominance persistence
        df['dominance_streak'] = df.groupby((df[self.result_col] != df[self.result_col].shift()).cumsum()).cumcount() + 1
        
        # Market regime classification
        df['market_phase'] = df[self.result_col].map({
            -1: 'Bear_Dominance',
            0: 'Congestion',
            1: 'Bull_Dominance'
        }).fillna('Undefined')
        
        result_columns = [
            self.result_col, 'dominance_strength', 'dominance_confidence',
            'dominance_score', 'dominance_streak', 'market_phase'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class DominanceStatusFeature(BaseAdvancedFeature):
    """
    Dominance Status Analysis Feature
    
    Tracks dominance status over time including duration,
    transitions, and temporal patterns.
    """
    
    def __init__(self,
                 dominance_col: str = 'dominance',
                 last_col: str = 'dominance_last',
                 duration_col: str = 'dominance_duration',
                 total_duration_col: str = 'dominance_duration_total',
                 time_col: str = 'dominance_time'):
        super().__init__('dominance_status')
        self.dominance_col = dominance_col
        self.last_col = last_col
        self.duration_col = duration_col
        self.total_duration_col = total_duration_col
        self.time_col = time_col
        self.required_columns = [dominance_col]
        
        # Store parameters
        self.parameters = {
            'dominance_col': dominance_col,
            'last_col': last_col,
            'duration_col': duration_col,
            'total_duration_col': total_duration_col,
            'time_col': time_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.dominance_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dominance status and temporal features"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.dominance_col}")
        
        df = self.preprocess(df.copy())
        
        # Previous dominance state
        df[self.last_col] = df[self.dominance_col].shift(1).fillna(0)
        
        # Dominance transitions
        df['dominance_changed'] = (df[self.dominance_col] != df[self.last_col]).astype(int)
        df['dominance_transition_type'] = 'None'
        
        # Classify transition types
        bull_to_bear = (df[self.last_col] == 1) & (df[self.dominance_col] == -1)
        bear_to_bull = (df[self.last_col] == -1) & (df[self.dominance_col] == 1)
        to_congestion = (df[self.last_col] != 0) & (df[self.dominance_col] == 0)
        from_congestion = (df[self.last_col] == 0) & (df[self.dominance_col] != 0)
        
        df.loc[bull_to_bear, 'dominance_transition_type'] = 'Bull_to_Bear'
        df.loc[bear_to_bull, 'dominance_transition_type'] = 'Bear_to_Bull'
        df.loc[to_congestion, 'dominance_transition_type'] = 'To_Congestion'
        df.loc[from_congestion, 'dominance_transition_type'] = 'From_Congestion'
        
        # Duration in current state
        groups = (df[self.dominance_col] != df[self.dominance_col].shift()).cumsum()
        df[self.duration_col] = df.groupby(groups).cumcount() + 1
        
        # Total duration tracking (simplified implementation)
        df[self.total_duration_col] = df[self.duration_col]  # For compatibility
        df[self.time_col] = df[self.duration_col]  # For compatibility
        
        # Rolling statistics for different regimes
        window = 168  # 1 week
        df['bull_time_ratio'] = (df[self.dominance_col] == 1).rolling(window=window).mean()
        df['bear_time_ratio'] = (df[self.dominance_col] == -1).rolling(window=window).mean()
        df['congestion_time_ratio'] = (df[self.dominance_col] == 0).rolling(window=window).mean()
        
        # Transition frequency
        df['transition_frequency'] = df['dominance_changed'].rolling(window=24).sum()
        
        # Regime stability
        df['regime_stability'] = df[self.dominance_col].rolling(window=24).std()
        
        # Average regime duration
        df['avg_regime_duration'] = df[self.duration_col].rolling(window=24).mean()
        
        # Dominant regime in recent period
        recent_window = 72  # 3 days
        recent_bull = (df[self.dominance_col] == 1).rolling(window=recent_window).sum()
        recent_bear = (df[self.dominance_col] == -1).rolling(window=recent_window).sum()
        recent_cong = (df[self.dominance_col] == 0).rolling(window=recent_window).sum()
        
        conditions = [
            recent_bull > recent_bear,
            recent_bear > recent_bull,
            recent_cong >= np.maximum(recent_bull, recent_bear)
        ]
        choices = ['Recent_Bull', 'Recent_Bear', 'Recent_Congestion']
        df['recent_dominant_regime'] = np.select(conditions, choices, default='Mixed')
        
        result_columns = [
            self.last_col, self.duration_col, self.total_duration_col, self.time_col,
            'dominance_changed', 'dominance_transition_type', 'bull_time_ratio',
            'bear_time_ratio', 'congestion_time_ratio', 'transition_frequency',
            'regime_stability', 'avg_regime_duration', 'recent_dominant_regime'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class DominanceClassificationFeature(BaseAdvancedFeature):
    """
    Dominance Classification Feature
    
    Classifies dominance patterns and provides additional
    strength and persistence indicators.
    """
    
    def __init__(self,
                 dominance_col: str = 'dominance'):
        super().__init__('dominance_classification')
        self.dominance_col = dominance_col
        self.required_columns = [dominance_col]
        
        # Store parameters
        self.parameters = {
            'dominance_col': dominance_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.dominance_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify dominance patterns and strength"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.dominance_col}")
        
        df = self.preprocess(df.copy())
        
        # Basic dominance classification
        df['dominance_class'] = df[self.dominance_col].map({
            -1: 'Bear',
            0: 'Congestion',
            1: 'Bull'
        }).fillna('Congestion')
        
        # Keep/strengthen analysis
        df['is_keep'] = ((df[self.dominance_col] == df[self.dominance_col].shift(1)) & 
                        (df[self.dominance_col] != 0)).astype(int)
        
        # Strengthening detection (basic implementation)
        df['is_strengthen'] = 0  # Placeholder for more complex logic
        
        # Enhanced classification with strength levels
        conditions = [
            df[self.dominance_col] == -1,
            df[self.dominance_col] == 0,
            df[self.dominance_col] == 1
        ]
        choices = ['Bear_Dominance', 'Market_Congestion', 'Bull_Dominance']
        df['detailed_dominance_class'] = np.select(conditions, choices, default='Undefined')
        
        # Dominance momentum
        df['dominance_momentum'] = df[self.dominance_col].diff()
        df['dominance_acceleration'] = df['dominance_momentum'].diff()
        
        # Persistence scoring
        persistence_window = 12
        df['dominance_persistence_score'] = df[self.dominance_col].rolling(
            window=persistence_window
        ).apply(lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0, raw=False)
        
        # Volatility of dominance
        df['dominance_volatility'] = df[self.dominance_col].rolling(window=24).std()
        
        # Trend strength
        trend_window = 24
        df['dominance_trend_strength'] = df[self.dominance_col].rolling(
            window=trend_window
        ).apply(lambda x: np.abs(x.mean()) if len(x) > 0 else 0, raw=False)
        
        # Quality metrics
        df['dominance_quality'] = df['dominance_persistence_score'] * df['dominance_trend_strength']
        
        # Signal confidence
        df['dominance_confidence'] = np.where(
            df[self.dominance_col] == 0,
            0,  # No confidence in congestion
            np.minimum(1.0, df['dominance_quality'] * 2)  # Scale to 0-1
        )
        
        # Regime change probability
        regime_change_features = [
            df['dominance_volatility'],
            1 - df['dominance_persistence_score'],
            np.abs(df['dominance_acceleration'])
        ]
        df['regime_change_probability'] = np.mean(regime_change_features, axis=0)
        
        result_columns = [
            'dominance_class', 'is_keep', 'is_strengthen', 'detailed_dominance_class',
            'dominance_momentum', 'dominance_acceleration', 'dominance_persistence_score',
            'dominance_volatility', 'dominance_trend_strength', 'dominance_quality',
            'dominance_confidence', 'regime_change_probability'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class MarketRegimeFeature(BaseAdvancedFeature):
    """
    Market Regime Analysis Feature
    
    Advanced market regime detection and classification
    based on dominance patterns and market dynamics.
    """
    
    def __init__(self,
                 dominance_col: str = 'dominance',
                 bin_col: str = 'bin_index',
                 rpn_col: str = 'risk_priority_number'):
        super().__init__('market_regime')
        self.dominance_col = dominance_col
        self.bin_col = bin_col
        self.rpn_col = rpn_col
        self.required_columns = [dominance_col]
        
        # Store parameters
        self.parameters = {
            'dominance_col': dominance_col,
            'bin_col': bin_col,
            'rpn_col': rpn_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.dominance_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze market regimes and dynamics"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.dominance_col}")
        
        df = self.preprocess(df.copy())
        
        # Market regime based on dominance patterns
        window = 72  # 3 days
        bull_ratio = (df[self.dominance_col] == 1).rolling(window=window).mean()
        bear_ratio = (df[self.dominance_col] == -1).rolling(window=window).mean()
        cong_ratio = (df[self.dominance_col] == 0).rolling(window=window).mean()
        
        # Regime classification
        conditions = [
            bull_ratio > 0.6,
            bear_ratio > 0.6,
            cong_ratio > 0.6,
            (bull_ratio > 0.4) & (bear_ratio < 0.3),
            (bear_ratio > 0.4) & (bull_ratio < 0.3),
            True  # Default case
        ]
        choices = [
            'Strong_Bull', 'Strong_Bear', 'Congestion',
            'Weak_Bull', 'Weak_Bear', 'Mixed'
        ]
        df['market_regime'] = np.select(conditions, choices, default='Mixed')
        
        # Regime volatility
        df['regime_volatility'] = df[self.dominance_col].rolling(window=24).std()
        
        # Regime transitions
        df['regime_changed'] = (df['market_regime'] != df['market_regime'].shift(1)).astype(int)
        df['regime_duration'] = df.groupby(
            (df['market_regime'] != df['market_regime'].shift()).cumsum()
        ).cumcount() + 1
        
        # Market stress indicators
        if self.bin_col in df.columns:
            # Extreme conditions
            df['extreme_conditions'] = ((df[self.bin_col] <= 1) | (df[self.bin_col] >= 7)).astype(int)
            df['extreme_ratio'] = df['extreme_conditions'].rolling(window=24).mean()
            
            # Market balance
            df['market_balance'] = 4 - np.abs(df[self.bin_col] - 4)  # Distance from center
            df['market_imbalance'] = df['market_balance'].rolling(window=24).std()
        
        # Regime strength
        df['regime_strength'] = np.maximum(bull_ratio, bear_ratio)
        df['regime_conviction'] = np.abs(bull_ratio - bear_ratio)
        
        # Market efficiency (lower volatility = higher efficiency)
        df['market_efficiency'] = 1 / (1 + df['regime_volatility'])
        
        # Trend consistency
        trend_window = 48
        df['trend_consistency'] = df[self.dominance_col].rolling(
            window=trend_window
        ).apply(lambda x: np.abs(x.mean()) / (x.std() + 1e-10) if len(x) > 0 else 0, raw=False)
        
        result_columns = [
            'market_regime', 'regime_volatility', 'regime_changed',
            'regime_duration', 'regime_strength', 'regime_conviction',
            'market_efficiency', 'trend_consistency'
        ]
        
        # Add conditional columns if available
        if self.bin_col in df.columns:
            result_columns.extend(['extreme_conditions', 'extreme_ratio', 'market_balance', 'market_imbalance'])
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df