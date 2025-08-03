"""
Trading Signal Features

This module implements trading signal generation features including
ceiling/bottom detection, reversal signals, and beta analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from .base import BaseAdvancedFeature


class BetaCalculationFeature(BaseAdvancedFeature):
    """
    Beta Calculation Feature
    
    Calculates regression coefficient (beta) for trend analysis
    and signal generation.
    """
    
    def __init__(self,
                 input_col: str = 'diff_ls_cwt_kf',
                 result_col: str = 'diff_beta_8',
                 window_size: int = 8):
        super().__init__('beta_calculation')
        self.input_col = input_col
        self.result_col = result_col
        self.window_size = window_size
        self.required_columns = [input_col]
        
        # Store parameters
        self.parameters = {
            'input_col': input_col,
            'result_col': result_col,
            'window_size': window_size
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.input_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate beta (regression coefficient)"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.input_col}")
        
        df = self.preprocess(df.copy())
        
        try:
            # Calculate rolling beta using least squares regression
            def calculate_beta(series):
                n = len(series)
                if n < 2:
                    return 0
                
                x = np.arange(n)
                y = series.values
                
                # Linear regression: y = a + bx, where b is beta
                x_mean = x.mean()
                y_mean = y.mean()
                
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                
                return numerator / (denominator + 1e-10)
            
            df[self.result_col] = df[self.input_col].rolling(
                window=self.window_size, min_periods=2
            ).apply(calculate_beta, raw=False)
            df[self.result_col] = df[self.result_col].fillna(0)
            
            # Additional beta metrics
            df[f'{self.result_col}_ma'] = df[self.result_col].rolling(window=6).mean()
            df[f'{self.result_col}_std'] = df[self.result_col].rolling(window=12).std()
            df[f'{self.result_col}_zscore'] = ((df[self.result_col] - df[f'{self.result_col}_ma']) / 
                                              (df[f'{self.result_col}_std'] + 1e-10))
            
            # Beta direction and strength
            df[f'{self.result_col}_direction'] = np.sign(df[self.result_col])
            df[f'{self.result_col}_strength'] = np.abs(df[self.result_col])
            
            # Beta momentum
            df[f'{self.result_col}_momentum'] = df[self.result_col].diff()
            df[f'{self.result_col}_acceleration'] = df[f'{self.result_col}_momentum'].diff()
            
            result_columns = [
                self.result_col, f'{self.result_col}_ma', f'{self.result_col}_std',
                f'{self.result_col}_zscore', f'{self.result_col}_direction',
                f'{self.result_col}_strength', f'{self.result_col}_momentum',
                f'{self.result_col}_acceleration'
            ]
            
        except Exception as e:
            self.logger.error(f"Error calculating beta: {e}")
            df[self.result_col] = 0
            result_columns = [self.result_col]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class CeilingBottomFeature(BaseAdvancedFeature):
    """
    Ceiling/Bottom Detection Feature
    
    Detects when market hits ceiling or bottom conditions
    based on dominance and bin patterns.
    """
    
    def __init__(self,
                 dominance_col: str = 'dominance',
                 bin_col: str = 'bin_index',
                 diff_col: str = 'diff_ls_cwt_kf',
                 fll_col: str = 'fll_cwt_kf',
                 fsl_col: str = 'fsl_cwt_kf',
                 result_col: str = 'hit_ceiling_bottom'):
        super().__init__('ceiling_bottom')
        self.dominance_col = dominance_col
        self.bin_col = bin_col
        self.diff_col = diff_col
        self.fll_col = fll_col
        self.fsl_col = fsl_col
        self.result_col = result_col
        self.required_columns = [dominance_col, bin_col]
        
        # Store parameters
        self.parameters = {
            'dominance_col': dominance_col,
            'bin_col': bin_col,
            'diff_col': diff_col,
            'fll_col': fll_col,
            'fsl_col': fsl_col,
            'result_col': result_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect ceiling and bottom hits"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        
        # Initialize result
        df[self.result_col] = 0
        
        # Basic ceiling/bottom logic
        # Ceiling: High bins + Bear dominance
        ceiling_condition = (df[self.bin_col] >= 7) & (df[self.dominance_col] == -1)
        df.loc[ceiling_condition, self.result_col] = -1
        
        # Bottom: Low bins + Bull dominance
        bottom_condition = (df[self.bin_col] <= 1) & (df[self.dominance_col] == 1)
        df.loc[bottom_condition, self.result_col] = 1
        
        # Enhanced conditions with liquidation differentials
        if self.diff_col in df.columns:
            # Strengthen ceiling detection with negative diff (FSL > FLL)
            enhanced_ceiling = ceiling_condition & (df[self.diff_col] < -0.1)
            df.loc[enhanced_ceiling, self.result_col] = -1
            
            # Strengthen bottom detection with positive diff (FLL > FSL)
            enhanced_bottom = bottom_condition & (df[self.diff_col] > 0.1)
            df.loc[enhanced_bottom, self.result_col] = 1
        
        # Ceiling/bottom strength
        df['cb_strength'] = np.abs(df[self.result_col])
        
        # Time since last ceiling/bottom
        df['time_since_ceiling'] = 0
        df['time_since_bottom'] = 0
        
        ceiling_indices = df[df[self.result_col] == -1].index
        bottom_indices = df[df[self.result_col] == 1].index
        
        for i in df.index:
            # Time since last ceiling
            recent_ceilings = ceiling_indices[ceiling_indices <= i]
            if len(recent_ceilings) > 0:
                df.loc[i, 'time_since_ceiling'] = i - recent_ceilings[-1]
            else:
                df.loc[i, 'time_since_ceiling'] = 999  # No recent ceiling
            
            # Time since last bottom
            recent_bottoms = bottom_indices[bottom_indices <= i]
            if len(recent_bottoms) > 0:
                df.loc[i, 'time_since_bottom'] = i - recent_bottoms[-1]
            else:
                df.loc[i, 'time_since_bottom'] = 999  # No recent bottom
        
        # Ceiling/bottom frequency
        df['ceiling_frequency'] = (df[self.result_col] == -1).rolling(window=168).sum()
        df['bottom_frequency'] = (df[self.result_col] == 1).rolling(window=168).sum()
        
        # Market position relative to recent extremes
        df['position_from_ceiling'] = np.minimum(df['time_since_ceiling'], 48) / 48
        df['position_from_bottom'] = np.minimum(df['time_since_bottom'], 48) / 48
        
        # Extreme market stress indicator
        df['extreme_market_stress'] = (
            (df['ceiling_frequency'] > 5) | (df['bottom_frequency'] > 5)
        ).astype(int)
        
        result_columns = [
            self.result_col, 'cb_strength', 'time_since_ceiling', 'time_since_bottom',
            'ceiling_frequency', 'bottom_frequency', 'position_from_ceiling',
            'position_from_bottom', 'extreme_market_stress'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class ReversalSignalFeature(BaseAdvancedFeature):
    """
    Reversal Signal Feature
    
    Detects reversal signals based on ceiling/bottom hits
    and beta analysis.
    """
    
    def __init__(self,
                 cb_col: str = 'hit_ceiling_bottom',
                 beta_col: str = 'diff_beta_8',
                 diff_col: str = 'diff_ls_cwt_kf',
                 result_col: str = 'reverse_ceiling_bottom',
                 window_size: int = 8):
        super().__init__('reversal_signal')
        self.cb_col = cb_col
        self.beta_col = beta_col
        self.diff_col = diff_col
        self.result_col = result_col
        self.window_size = window_size
        self.required_columns = [cb_col, diff_col]
        
        # Store parameters
        self.parameters = {
            'cb_col': cb_col,
            'beta_col': beta_col,
            'diff_col': diff_col,
            'result_col': result_col,
            'window_size': window_size
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect reversal signals"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        
        # Ensure beta column exists
        if self.beta_col not in df.columns:
            # Calculate beta inline if missing
            beta_feature = BetaCalculationFeature(
                input_col=self.diff_col,
                result_col=self.beta_col,
                window_size=self.window_size
            )
            df = beta_feature.calculate(df)
        
        # Initialize result
        df[self.result_col] = 0
        
        # Reversal logic
        # Reverse ceiling hits when beta turns positive (upward momentum after ceiling)
        ceiling_reversal = (
            (df[self.cb_col].shift(self.window_size) == -1) & 
            (df[self.beta_col] > 0.01)  # Positive momentum threshold
        )
        df.loc[ceiling_reversal, self.result_col] = 1
        
        # Reverse bottom hits when beta turns negative (downward momentum after bottom)
        bottom_reversal = (
            (df[self.cb_col].shift(self.window_size) == 1) & 
            (df[self.beta_col] < -0.01)  # Negative momentum threshold
        )
        df.loc[bottom_reversal, self.result_col] = -1
        
        # Enhanced reversal detection
        # Look for momentum divergence
        recent_ceiling = df[self.cb_col].rolling(window=24).min() == -1
        recent_bottom = df[self.cb_col].rolling(window=24).max() == 1
        
        # Strong reversal signals
        strong_bull_reversal = (
            recent_ceiling & 
            (df[self.beta_col] > 0.05) &
            (df[self.diff_col] > df[self.diff_col].shift(6))
        )
        df.loc[strong_bull_reversal, self.result_col] = 1
        
        strong_bear_reversal = (
            recent_bottom & 
            (df[self.beta_col] < -0.05) &
            (df[self.diff_col] < df[self.diff_col].shift(6))
        )
        df.loc[strong_bear_reversal, self.result_col] = -1
        
        # Reversal strength
        df['reversal_strength'] = np.abs(df[self.result_col])
        
        # Reversal confidence based on beta strength
        df['reversal_confidence'] = np.minimum(1.0, np.abs(df[self.beta_col]) * 10)
        
        # Time between reversals
        df['reversal_frequency'] = np.abs(df[self.result_col]).rolling(window=168).sum()
        
        # Reversal success tracking (simplified)
        df['reversal_type'] = df[self.result_col].map({
            -1: 'Bear_Reversal',
            0: 'No_Reversal', 
            1: 'Bull_Reversal'
        }).fillna('No_Reversal')
        
        # Market turning point indicator
        df['major_turning_point'] = (
            (df['reversal_strength'] > 0) & 
            (df['reversal_confidence'] > 0.5)
        ).astype(int)
        
        result_columns = [
            self.result_col, 'reversal_strength', 'reversal_confidence',
            'reversal_frequency', 'reversal_type', 'major_turning_point'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class SignalGenerationFeature(BaseAdvancedFeature):
    """
    Trading Signal Generation Feature
    
    Generates multi-level trading signals based on dominance,
    ceiling/bottom hits, and reversal patterns.
    """
    
    def __init__(self,
                 dominance_col: str = 'dominance',
                 cb_col: str = 'hit_ceiling_bottom',
                 reversal_col: str = 'reverse_ceiling_bottom'):
        super().__init__('signal_generation')
        self.dominance_col = dominance_col
        self.cb_col = cb_col
        self.reversal_col = reversal_col
        self.required_columns = [dominance_col, cb_col, reversal_col]
        
        # Store parameters
        self.parameters = {
            'dominance_col': dominance_col,
            'cb_col': cb_col,
            'reversal_col': reversal_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-level trading signals"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        
        # Level 1: Basic dominance signals
        df['signal_l1'] = df[self.dominance_col]
        
        # Level 2: Ceiling/bottom signals
        df['signal_l2'] = df[self.cb_col]
        
        # Level 3: Reversal signals
        df['signal_l3'] = df[self.reversal_col]
        
        # Combined signal strength
        df['signal_strength'] = (
            np.abs(df['signal_l1']) * 0.3 +
            np.abs(df['signal_l2']) * 0.4 +
            np.abs(df['signal_l3']) * 0.3
        )
        
        # Signal direction consistency
        signals = [df['signal_l1'], df['signal_l2'], df['signal_l3']]
        df['signal_consistency'] = np.sum([
            (sig > 0).astype(int) for sig in signals
        ], axis=0) - np.sum([
            (sig < 0).astype(int) for sig in signals
        ], axis=0)
        
        # Overall signal classification
        def classify_signal(row):
            if row['signal_l3'] != 0:
                return 'Strong_Buy' if row['signal_l3'] > 0 else 'Strong_Sell'
            elif row['signal_l2'] != 0:
                return 'Buy' if row['signal_l2'] > 0 else 'Sell'
            elif row['signal_l1'] != 0:
                return 'Weak_Buy' if row['signal_l1'] > 0 else 'Weak_Sell'
            else:
                return 'Neutral'
        
        df['signal_class'] = df.apply(classify_signal, axis=1)
        
        # Signal quality metrics
        df['signal_quality'] = df['signal_strength'] * np.abs(df['signal_consistency']) / 3
        
        # Signal persistence
        df['signal_persistence'] = df['signal_class'].rolling(window=6).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0, raw=False
        )
        
        # Entry/exit signals
        df['entry_signal'] = ((df['signal_strength'] > 0.5) & 
                             (df['signal_quality'] > 0.3)).astype(int)
        
        df['exit_signal'] = ((df['signal_strength'] < 0.2) |
                            (df['signal_class'] == 'Neutral')).astype(int)
        
        # Signal timing
        df['signal_timing_score'] = (
            df['signal_strength'] * 0.4 +
            df['signal_quality'] * 0.3 +
            df['signal_persistence'] * 0.3
        )
        
        # Risk assessment
        conditions = [
            df['signal_timing_score'] > 0.7,
            df['signal_timing_score'] > 0.4,
            df['signal_timing_score'] > 0.2,
            True
        ]
        choices = ['Low_Risk', 'Medium_Risk', 'High_Risk', 'Very_High_Risk']
        df['signal_risk'] = np.select(conditions, choices, default='Very_High_Risk')
        
        result_columns = [
            'signal_l1', 'signal_l2', 'signal_l3', 'signal_class',
            'signal_strength', 'signal_consistency', 'signal_quality',
            'signal_persistence', 'entry_signal', 'exit_signal',
            'signal_timing_score', 'signal_risk'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class SignalAnalysisFeature(BaseAdvancedFeature):
    """
    Signal Analysis and Performance Feature
    
    Analyzes signal patterns, performance, and provides
    additional signal intelligence.
    """
    
    def __init__(self,
                 signal_class_col: str = 'signal_class',
                 price_col: str = 'price'):
        super().__init__('signal_analysis')
        self.signal_class_col = signal_class_col
        self.price_col = price_col
        self.required_columns = [signal_class_col]
        
        # Store parameters
        self.parameters = {
            'signal_class_col': signal_class_col,
            'price_col': price_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.signal_class_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze signal patterns and performance"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.signal_class_col}")
        
        df = self.preprocess(df.copy())
        
        # Signal frequency analysis
        signal_types = ['Strong_Buy', 'Buy', 'Weak_Buy', 'Strong_Sell', 'Sell', 'Weak_Sell']
        
        for signal_type in signal_types:
            df[f'{signal_type.lower()}_frequency'] = (
                df[self.signal_class_col] == signal_type
            ).rolling(window=168).sum()
        
        # Overall signal activity
        df['signal_activity'] = (df[self.signal_class_col] != 'Neutral').rolling(window=24).sum()
        
        # Signal transitions
        df['signal_changed'] = (df[self.signal_class_col] != df[self.signal_class_col].shift(1)).astype(int)
        df['signal_transition_frequency'] = df['signal_changed'].rolling(window=24).sum()
        
        # Time in signals
        df['time_in_current_signal'] = df.groupby(
            (df[self.signal_class_col] != df[self.signal_class_col].shift()).cumsum()
        ).cumcount() + 1
        
        # Signal clustering (periods of high activity)
        df['signal_cluster'] = (df['signal_activity'] > df['signal_activity'].rolling(window=168).quantile(0.8)).astype(int)
        
        # Performance tracking (if price available)
        if self.price_col in df.columns:
            # Forward returns for signal evaluation
            for horizon in [1, 6, 24]:
                df[f'forward_return_{horizon}h'] = (
                    df[self.price_col].shift(-horizon) / df[self.price_col] - 1
                ) * 100
            
            # Signal accuracy indicators
            buy_signals = df[self.signal_class_col].str.contains('Buy', na=False)
            sell_signals = df[self.signal_class_col].str.contains('Sell', na=False)
            
            df['signal_direction_correct_1h'] = (
                (buy_signals & (df['forward_return_1h'] > 0)) |
                (sell_signals & (df['forward_return_1h'] < 0))
            ).astype(int)
            
            df['signal_direction_correct_6h'] = (
                (buy_signals & (df['forward_return_6h'] > 0)) |
                (sell_signals & (df['forward_return_6h'] < 0))
            ).astype(int)
            
            # Rolling accuracy
            df['signal_accuracy_1h'] = df['signal_direction_correct_1h'].rolling(window=24).mean()
            df['signal_accuracy_6h'] = df['signal_direction_correct_6h'].rolling(window=24).mean()
        
        # Market condition when signals occur
        df['signal_in_extreme_conditions'] = (
            df['signal_activity'] > 0
        ).astype(int)  # Placeholder for more complex logic
        
        # Signal quality score
        base_columns = [
            'signal_activity', 'time_in_current_signal'
        ]
        
        if self.price_col in df.columns:
            base_columns.extend(['signal_accuracy_1h', 'signal_accuracy_6h'])
        
        # Normalize and combine available metrics
        quality_components = []
        for col in base_columns:
            if col in df.columns:
                normalized = (df[col] - df[col].rolling(window=168).min()) / (
                    df[col].rolling(window=168).max() - df[col].rolling(window=168).min() + 1e-10
                )
                quality_components.append(normalized)
        
        if quality_components:
            df['signal_overall_quality'] = np.mean(quality_components, axis=0)
        else:
            df['signal_overall_quality'] = 0.5  # Neutral default
        
        result_columns = [
            'signal_activity', 'signal_changed', 'signal_transition_frequency',
            'time_in_current_signal', 'signal_cluster', 'signal_in_extreme_conditions',
            'signal_overall_quality'
        ]
        
        # Add frequency columns
        for signal_type in signal_types:
            col_name = f'{signal_type.lower()}_frequency'
            if col_name in df.columns:
                result_columns.append(col_name)
        
        # Add performance columns if available
        if self.price_col in df.columns:
            result_columns.extend([
                'forward_return_1h', 'forward_return_6h', 'forward_return_24h',
                'signal_direction_correct_1h', 'signal_direction_correct_6h',
                'signal_accuracy_1h', 'signal_accuracy_6h'
            ])
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df