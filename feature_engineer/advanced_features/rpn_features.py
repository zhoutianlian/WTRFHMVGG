"""
Risk Priority Number (RPN) Features

This module implements the core RPN features including CWT detrending,
Kalman filtering, and correlation analysis as the foundational layer.
"""

import pandas as pd
import numpy as np
import pywt
from pykalman import KalmanFilter
from typing import List, Optional, Literal
from .base import BaseAdvancedFeature


class CWTDetrendingFeature(BaseAdvancedFeature):
    """
    Continuous Wavelet Transform (CWT) detrending feature
    
    Applies CWT detrending to remove noise and extract underlying trends
    from liquidation dominance and futures data.
    """
    
    def __init__(self,
                 input_columns: List[str] = ['lld_normal', 'fll_normal', 'fsl_normal'],
                 wavelet: str = 'coif3',
                 level: int = 6):
        super().__init__('cwt_detrending')
        self.input_columns = input_columns
        self.wavelet = wavelet
        self.level = level
        self.required_columns = input_columns
        
        # Store parameters
        self.parameters = {
            'input_columns': input_columns,
            'wavelet': wavelet,
            'level': level
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply CWT detrending to specified columns"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        result_columns = []
        
        for col in self.input_columns:
            if col not in df.columns:
                continue
                
            output_col = col.replace('_normal', '_cwt')
            result_columns.append(output_col)
            
            # Fill NaN values
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            try:
                if len(df[col].dropna()) < 2**self.level:
                    self.logger.warning(f"Insufficient data for CWT level {self.level}, using original values")
                    df[output_col] = df[col]
                    continue
                
                # Apply CWT decomposition
                coeffs = pywt.wavedec(df[col].values, self.wavelet, level=self.level)
                approx = coeffs[0]
                
                # Reconstruct trend
                trend = pywt.upcoef('a', approx, self.wavelet, level=self.level, take=len(df[col]))
                df[output_col] = trend[:len(df[col])]
                
            except Exception as e:
                self.logger.warning(f"Error in CWT detrending for {col}: {e}, using original values")
                df[output_col] = df[col]  # Fallback to original
        
        # Calculate diff_ls_cwt if both fll and fsl are processed
        if 'fll_cwt' in df.columns and 'fsl_cwt' in df.columns:
            df['diff_ls_cwt'] = df['fll_cwt'] - df['fsl_cwt']
            result_columns.append('diff_ls_cwt')
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class KalmanFilteringFeature(BaseAdvancedFeature):
    """
    Kalman Filter smoothing feature
    
    Applies Kalman filtering to CWT-detrended data for additional smoothing
    and noise reduction.
    """
    
    def __init__(self,
                 input_columns: List[str] = ['lld_cwt', 'fll_cwt', 'fsl_cwt'],
                 initial_state_mean: float = 0.49,
                 initial_state_covariance: float = 50,
                 observation_covariance: float = 80,
                 transition_covariance: float = 0.05):
        super().__init__('kalman_filtering')
        self.input_columns = input_columns
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance
        self.required_columns = input_columns
        
        # Store parameters
        self.parameters = {
            'input_columns': input_columns,
            'initial_state_mean': initial_state_mean,
            'initial_state_covariance': initial_state_covariance,
            'observation_covariance': observation_covariance,
            'transition_covariance': transition_covariance
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return any(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Kalman filtering to CWT-detrended data"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        result_columns = []
        
        # Initialize Kalman filter
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=self.initial_state_mean,
            initial_state_covariance=self.initial_state_covariance,
            observation_covariance=self.observation_covariance,
            transition_covariance=self.transition_covariance
        )
        
        for col in self.input_columns:
            if col not in df.columns:
                continue
                
            output_col = col + '_kf'
            result_columns.append(output_col)
            
            try:
                valid_data = df[col].dropna()
                if len(valid_data) < 2:
                    df[output_col] = df[col]
                    continue
                
                # Apply Kalman filtering
                filtered_state_means, _ = kf.filter(valid_data.values)
                
                # Map back to original indices
                df[output_col] = np.nan
                df.loc[valid_data.index, output_col] = filtered_state_means.flatten()
                
                # Fill remaining NaN values
                df[output_col] = df[output_col].fillna(method='ffill').fillna(method='bfill')
                
            except Exception as e:
                self.logger.warning(f"Error in Kalman filtering for {col}: {e}, using original values")
                df[output_col] = df[col]  # Fallback
        
        # Calculate diff_ls_cwt_kf if both components exist
        if 'fll_cwt_kf' in df.columns and 'fsl_cwt_kf' in df.columns:
            df['diff_ls_cwt_kf'] = df['fll_cwt_kf'] - df['fsl_cwt_kf']
            result_columns.append('diff_ls_cwt_kf')
        elif 'diff_ls_cwt' in df.columns:
            # Apply Kalman to diff_ls_cwt if it exists
            try:
                valid_data = df['diff_ls_cwt'].dropna()
                if len(valid_data) >= 2:
                    filtered_state_means, _ = kf.filter(valid_data.values)
                    df['diff_ls_cwt_kf'] = np.nan
                    df.loc[valid_data.index, 'diff_ls_cwt_kf'] = filtered_state_means.flatten()
                    df['diff_ls_cwt_kf'] = df['diff_ls_cwt_kf'].fillna(method='ffill').fillna(method='bfill')
                    result_columns.append('diff_ls_cwt_kf')
            except Exception as e:
                self.logger.warning(f"Error in Kalman filtering for diff_ls_cwt: {e}")
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class EMAFeature(BaseAdvancedFeature):
    """
    Exponential Moving Average feature
    
    Applies EMA smoothing to Kalman-filtered data for final smoothing.
    """
    
    def __init__(self,
                 input_configs: List[dict] = [
                     {'input': 'diff_ls_cwt_kf', 'output': 'diff_ls_smooth', 'span': 16},
                     {'input': 'lld_cwt_kf', 'output': 'lld_cwt_kf_smooth', 'span': 12}
                 ]):
        super().__init__('ema_smoothing')
        self.input_configs = input_configs
        self.required_columns = [config['input'] for config in input_configs]
        
        # Store parameters
        self.parameters = {
            'input_configs': input_configs
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return any(config['input'] in df.columns for config in self.input_configs)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply EMA smoothing to specified columns"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        result_columns = []
        
        for config in self.input_configs:
            input_col = config['input']
            output_col = config['output']
            span = config['span']
            
            if input_col in df.columns:
                df[output_col] = df[input_col].ewm(span=span, adjust=False).mean()
                result_columns.append(output_col)
            else:
                self.logger.warning(f"Column {input_col} not found for EMA calculation")
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class RPNExtremeFeature(BaseAdvancedFeature):
    """
    RPN Extreme Detection Feature
    
    Detects extreme conditions in the Risk Priority Number based on
    historical patterns and sequences.
    """
    
    def __init__(self,
                 rpn_col: str = 'lld_cwt_kf',
                 window_size: int = 30 * 24,  # 30 days in hours
                 result_col: str = 'is_rpn_extreme'):
        super().__init__('rpn_extreme')
        self.rpn_col = rpn_col
        self.window_size = window_size
        self.result_col = result_col
        self.required_columns = [rpn_col]
        
        # Store parameters
        self.parameters = {
            'rpn_col': rpn_col,
            'window_size': window_size,
            'result_col': result_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.rpn_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate extreme RPN indicator"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.rpn_col}")
        
        df = self.preprocess(df.copy())
        
        def extreme_detection(rolling_rpn):
            """Detect extreme conditions based on original algorithm"""
            try:
                rolling_rpn = rolling_rpn.tolist()
                if 8 in rolling_rpn:
                    last_8_index = len(rolling_rpn) - 1 - rolling_rpn[::-1].index(8)
                    sublist = rolling_rpn[last_8_index + 1:]
                    return -1 if 4 not in sublist else 0
                if 0 in rolling_rpn:
                    last_0_index = len(rolling_rpn) - 1 - rolling_rpn[::-1].index(0)
                    sublist = rolling_rpn[last_0_index + 1:]
                    return 1 if 5 not in sublist else 0
                return 0
            except:
                return 0
        
        # Apply extreme detection
        df[self.result_col] = df[self.rpn_col].rolling(
            window=self.window_size, min_periods=1
        ).apply(extreme_detection, raw=False)
        df[self.result_col] = df[self.result_col].fillna(0)
        
        # Additional extreme statistics
        df[f'{self.result_col}_count'] = df[self.result_col].rolling(window=168).sum()  # Weekly count
        df[f'{self.result_col}_intensity'] = df[self.result_col].abs().rolling(window=24).mean()  # Daily intensity
        
        result_columns = [self.result_col, f'{self.result_col}_count', f'{self.result_col}_intensity']
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class CorrelationAnalysisFeature(BaseAdvancedFeature):
    """
    Correlation Analysis Feature
    
    Analyzes correlation patterns between FLL and FSL liquidations
    to classify market conditions.
    """
    
    def __init__(self,
                 fll_col: str = 'fll_cwt_kf',
                 fsl_col: str = 'fsl_cwt_kf'):
        super().__init__('correlation_analysis')
        self.fll_col = fll_col
        self.fsl_col = fsl_col
        self.required_columns = [fll_col, fsl_col]
        
        # Store parameters
        self.parameters = {
            'fll_col': fll_col,
            'fsl_col': fsl_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlation patterns between liquidations"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        
        # Calculate deltas
        df['delta_fll'] = df[self.fll_col].diff().fillna(0)
        df['delta_fsl'] = df[self.fsl_col].diff().fillna(0)
        
        # Classify correlation cases based on original logic
        conditions = [
            (df['delta_fll'] > 0) & (df['delta_fsl'] > 0) & (df[self.fll_col] > df[self.fsl_col]),
            (df['delta_fll'] > 0) & (df['delta_fsl'] > 0) & (df[self.fll_col] <= df[self.fsl_col]),
            (df['delta_fll'] < 0) & (df['delta_fsl'] < 0) & (df[self.fll_col] > df[self.fsl_col]),
            (df['delta_fll'] < 0) & (df['delta_fsl'] < 0) & (df[self.fll_col] <= df[self.fsl_col]),
            (df['delta_fll'] > 0) & (df['delta_fsl'] < 0) & (df[self.fll_col] > df[self.fsl_col]),
            (df['delta_fll'] > 0) & (df['delta_fsl'] < 0) & (df[self.fll_col] <= df[self.fsl_col]),
            (df['delta_fll'] < 0) & (df['delta_fsl'] > 0) & (df[self.fll_col] > df[self.fsl_col]),
            (df['delta_fll'] < 0) & (df['delta_fsl'] > 0) & (df[self.fll_col] <= df[self.fsl_col]),
        ]
        choices = ['LUSUL', 'LUSUS', 'LDSDL', 'LDSDS', 'LUSDL', 'LUSDS', 'LDSUL', 'LDSUS']
        
        df['corr_case'] = np.select(conditions, choices, default='None')
        
        # Additional correlation metrics
        df['fll_fsl_ratio'] = df[self.fll_col] / (df[self.fsl_col] + 1e-10)
        df['liquidation_balance'] = (df[self.fll_col] - df[self.fsl_col]) / (df[self.fll_col] + df[self.fsl_col] + 1e-10)
        
        # Rolling correlation
        window = 24
        df['fll_fsl_correlation'] = df[self.fll_col].rolling(window=window).corr(df[self.fsl_col])
        
        # Correlation strength classification
        df['correlation_strength'] = pd.cut(
            df['fll_fsl_correlation'].abs(),
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Weak', 'Moderate', 'Strong'],
            include_lowest=True
        ).astype(str)
        
        result_columns = [
            'delta_fll', 'delta_fsl', 'corr_case', 'fll_fsl_ratio', 
            'liquidation_balance', 'fll_fsl_correlation', 'correlation_strength'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class RPNCalculationFeature(BaseAdvancedFeature):
    """
    Risk Priority Number Calculation Feature
    
    Calculates the main RPN value that serves as the foundation
    for all subsequent analysis.
    """
    
    def __init__(self,
                 primary_col: str = 'lld_cwt_kf',
                 fallback_col: str = 'lld_normal',
                 result_col: str = 'risk_priority_number'):
        super().__init__('rpn_calculation')
        self.primary_col = primary_col
        self.fallback_col = fallback_col
        self.result_col = result_col
        self.required_columns = [primary_col, fallback_col]
        
        # Store parameters
        self.parameters = {
            'primary_col': primary_col,
            'fallback_col': fallback_col,
            'result_col': result_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return any(col in df.columns for col in self.required_columns)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the main Risk Priority Number"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required columns: {self.required_columns}")
        
        df = self.preprocess(df.copy())
        
        # Set risk_priority_number (main RPN metric)
        if self.primary_col in df.columns:
            df[self.result_col] = df[self.primary_col]
        elif self.fallback_col in df.columns:
            df[self.result_col] = df[self.fallback_col]
        else:
            # Final fallback calculation
            df[self.result_col] = 0.5  # Neutral value
        
        # Additional RPN statistics
        df[f'{self.result_col}_ma'] = df[self.result_col].rolling(window=24).mean()
        df[f'{self.result_col}_std'] = df[self.result_col].rolling(window=24).std()
        df[f'{self.result_col}_zscore'] = ((df[self.result_col] - df[f'{self.result_col}_ma']) / 
                                          (df[f'{self.result_col}_std'] + 1e-10))
        
        # RPN trend analysis
        df[f'{self.result_col}_trend'] = df[self.result_col].diff().rolling(window=6).mean()
        df[f'{self.result_col}_momentum'] = df[self.result_col].pct_change(periods=6)
        
        result_columns = [
            self.result_col, f'{self.result_col}_ma', f'{self.result_col}_std',
            f'{self.result_col}_zscore', f'{self.result_col}_trend', f'{self.result_col}_momentum'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df