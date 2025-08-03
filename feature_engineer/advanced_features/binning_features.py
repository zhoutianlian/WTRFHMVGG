"""
Binning and Clustering Features

This module implements K-means clustering and momentum analysis features
for binning the Risk Priority Number into discrete categories.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional
from .base import BaseAdvancedFeature


class KMeansBinningFeature(BaseAdvancedFeature):
    """
    K-Means Binning Feature
    
    Applies K-means clustering to the Risk Priority Number to create
    discrete bins representing different market conditions.
    """
    
    def __init__(self,
                 rpn_col: str = 'risk_priority_number',
                 n_clusters: int = 9,
                 random_state: int = 42,
                 result_col: str = 'bin_index'):
        super().__init__('kmeans_binning')
        self.rpn_col = rpn_col
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.result_col = result_col
        self.required_columns = [rpn_col]
        
        # Store parameters
        self.parameters = {
            'rpn_col': rpn_col,
            'n_clusters': n_clusters,
            'random_state': random_state,
            'result_col': result_col
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.rpn_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply K-means clustering to create bins"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.rpn_col}")
        
        df = self.preprocess(df.copy())
        
        try:
            valid_data = df[self.rpn_col].dropna()
            if len(valid_data) < self.n_clusters:
                self.logger.warning(f"Insufficient data for {self.n_clusters} clusters, using default bins")
                df[self.result_col] = 4  # Default middle bin
                return self.postprocess(df)
            
            # Prepare data for clustering
            rpn_data = np.array(valid_data).reshape(-1, 1)
            
            # Apply K-means clustering
            kmeans = KMeans(
                n_clusters=self.n_clusters, 
                random_state=self.random_state, 
                n_init=10
            ).fit(rpn_data)
            
            # Get predictions for all valid data
            predictions = kmeans.predict(rpn_data)
            
            # Sort cluster centers and create mapping
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(cluster_centers)
            
            # Map to sorted indices (0=lowest, n_clusters-1=highest)
            label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
            mapped_predictions = [label_map[pred] for pred in predictions]
            
            # Assign back to dataframe
            df[self.result_col] = 4  # Default for NaN values (middle bin)
            df.loc[valid_data.index, self.result_col] = mapped_predictions
            
            # Store cluster information for analysis
            df['bin_centers'] = np.nan
            for i, center in enumerate(cluster_centers[sorted_indices]):
                mask = df[self.result_col] == i
                df.loc[mask, 'bin_centers'] = center
            
            # Calculate bin statistics
            df['bin_distance'] = np.abs(df[self.rpn_col] - df['bin_centers'])
            df['bin_confidence'] = 1 / (1 + df['bin_distance'])  # Higher = more confident
            
            # Bin transition detection
            df['bin_changed'] = (df[self.result_col] != df[self.result_col].shift(1)).astype(int)
            df['bin_stability'] = df[self.result_col].rolling(window=24).std()  # Lower = more stable
            
            result_columns = [
                self.result_col, 'bin_centers', 'bin_distance', 
                'bin_confidence', 'bin_changed', 'bin_stability'
            ]
            
        except Exception as e:
            self.logger.error(f"Error in K-means binning: {e}, using default bins")
            df[self.result_col] = 4  # Default middle bin
            result_columns = [self.result_col]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df


class BinMomentumFeature(BaseAdvancedFeature):
    """
    Bin Momentum Analysis Feature
    
    Analyzes momentum patterns in the bin indices to detect
    trends and turning points in market conditions.
    """
    
    def __init__(self,
                 bin_col: str = 'bin_index',
                 momentum_col: str = 'bin_momentum',
                 turn_col: str = 'bin_turn',
                 direction_col: str = 'bin_mm_direction',
                 window_size: int = 24 * 30):  # 30 days
        super().__init__('bin_momentum')
        self.bin_col = bin_col
        self.momentum_col = momentum_col
        self.turn_col = turn_col
        self.direction_col = direction_col
        self.window_size = window_size
        self.required_columns = [bin_col]
        
        # Store parameters
        self.parameters = {
            'bin_col': bin_col,
            'momentum_col': momentum_col,
            'turn_col': turn_col,
            'direction_col': direction_col,
            'window_size': window_size
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.bin_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bin momentum and trend analysis"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.bin_col}")
        
        df = self.preprocess(df.copy())
        
        # Calculate momentum for each row
        momentum_results = []
        for i in range(len(df)):
            start_idx = max(0, i - self.window_size)
            window_data = df[self.bin_col].iloc[start_idx:i+1].tolist()
            momentum_results.append(self._calculate_continuous_trend(window_data))
        
        # Unpack results
        df[self.momentum_col] = [r[0] for r in momentum_results]
        df[self.turn_col] = [r[1] for r in momentum_results]
        df[self.direction_col] = [r[2] for r in momentum_results]
        
        # Additional momentum metrics
        df['bin_velocity'] = df[self.bin_col].diff()
        df['bin_acceleration'] = df['bin_velocity'].diff()
        
        # Momentum strength
        df['momentum_strength'] = np.abs(df[self.momentum_col])
        df['momentum_persistence'] = df[self.direction_col].rolling(window=6).apply(
            lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0, raw=False
        )
        
        # Trend classification
        df['trend_type'] = pd.cut(
            df[self.momentum_col],
            bins=[-np.inf, -2, -1, 1, 2, np.inf],
            labels=['Strong_Down', 'Weak_Down', 'Sideways', 'Weak_Up', 'Strong_Up'],
            include_lowest=True
        ).astype(str)
        
        # Turn point analysis
        df['turn_intensity'] = df[self.turn_col].rolling(window=12).sum()  # Turns in last 12 periods
        df['is_major_turn'] = (df[self.turn_col] >= 3).astype(int)
        
        result_columns = [
            self.momentum_col, self.turn_col, self.direction_col,
            'bin_velocity', 'bin_acceleration', 'momentum_strength',
            'momentum_persistence', 'trend_type', 'turn_intensity', 'is_major_turn'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df
    
    def _calculate_continuous_trend(self, arr_num: List[int]) -> Tuple[int, int, int]:
        """Calculate continuous trend from original algorithm"""
        if len(arr_num) < 2:
            return (0, 0, 0)
        
        try:
            # Remove neighboring duplicates
            arr_num_unique = []
            for num in arr_num:
                if not arr_num_unique or num != arr_num_unique[-1]:
                    arr_num_unique.append(num)
            
            if len(arr_num_unique) < 2:
                return (0, 0, 0)
            
            last_num = arr_num_unique[-1]
            prev_num = arr_num_unique[-2]
            direction = -1 if last_num < prev_num else (1 if last_num > prev_num else 0)
            
            tmp_list = [last_num]
            var = 0
            n_turn = 0
            
            if direction == -1:
                for num in reversed(arr_num_unique[:-1]):
                    max_num = max(tmp_list)
                    if num > tmp_list[-1] or num >= max_num - 1:
                        tmp_list.append(num)
                        if num >= max_num - 1 and tmp_list.count(max_num) == 1:
                            n_turn += 1
                    else:
                        break
                var = last_num - max(tmp_list)
            elif direction == 1:
                for num in reversed(arr_num_unique[:-1]):
                    min_num = min(tmp_list)
                    if num < tmp_list[-1] or num <= min_num + 1:
                        tmp_list.append(num)
                        if num <= min_num + 1:
                            n_turn += 1
                    else:
                        break
                var = last_num - min(tmp_list)
            
            return (var, max(0, n_turn - 2), direction)
            
        except Exception as e:
            return (0, 0, 0)


class BinAnalysisFeature(BaseAdvancedFeature):
    """
    Advanced Bin Analysis Feature
    
    Provides comprehensive analysis of bin patterns including
    distribution analysis, regime detection, and stability metrics.
    """
    
    def __init__(self,
                 bin_col: str = 'bin_index',
                 n_bins: int = 9):
        super().__init__('bin_analysis')
        self.bin_col = bin_col
        self.n_bins = n_bins
        self.required_columns = [bin_col]
        
        # Store parameters
        self.parameters = {
            'bin_col': bin_col,
            'n_bins': n_bins
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        return self.bin_col in df.columns
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced bin analysis metrics"""
        self.log_calculation_start()
        
        if not self.validate_input(df):
            raise ValueError(f"Missing required column: {self.bin_col}")
        
        df = self.preprocess(df.copy())
        
        # Bin distribution analysis
        df['bin_percentile'] = df[self.bin_col] / (self.n_bins - 1)  # Convert to 0-1 scale
        
        # Rolling bin statistics
        window = 168  # 1 week
        df['bin_mean'] = df[self.bin_col].rolling(window=window).mean()
        df['bin_std'] = df[self.bin_col].rolling(window=window).std()
        df['bin_skew'] = df[self.bin_col].rolling(window=window).skew()
        
        # Bin regime detection
        df['is_extreme_low'] = (df[self.bin_col] <= 1).astype(int)
        df['is_extreme_high'] = (df[self.bin_col] >= self.n_bins - 2).astype(int)
        df['is_middle_range'] = ((df[self.bin_col] >= 3) & (df[self.bin_col] <= 5)).astype(int)
        
        # Time in regime
        df['time_in_low'] = df['is_extreme_low'].rolling(window=24).sum()
        df['time_in_high'] = df['is_extreme_high'].rolling(window=24).sum()
        df['time_in_middle'] = df['is_middle_range'].rolling(window=24).sum()
        
        # Bin transition patterns
        df['bin_up_moves'] = ((df[self.bin_col] > df[self.bin_col].shift(1))).rolling(window=24).sum()
        df['bin_down_moves'] = ((df[self.bin_col] < df[self.bin_col].shift(1))).rolling(window=24).sum()
        df['bin_no_moves'] = ((df[self.bin_col] == df[self.bin_col].shift(1))).rolling(window=24).sum()
        
        # Volatility of bin movements
        df['bin_volatility'] = df[self.bin_col].rolling(window=24).std()
        df['bin_range'] = (df[self.bin_col].rolling(window=24).max() - 
                          df[self.bin_col].rolling(window=24).min())
        
        # Bin momentum classification
        conditions = [
            df['bin_mean'] <= 2,
            (df['bin_mean'] > 2) & (df['bin_mean'] <= 4),
            (df['bin_mean'] > 4) & (df['bin_mean'] <= 6),
            df['bin_mean'] > 6
        ]
        labels = ['Bearish', 'Weak_Bearish', 'Weak_Bullish', 'Bullish']
        df['bin_regime'] = pd.cut(
            df['bin_mean'], 
            bins=[0, 2, 4, 6, self.n_bins],
            labels=labels,
            include_lowest=True
        ).astype(str)
        
        # Bin persistence (how long in current bin)
        df['bin_persistence'] = df.groupby((df[self.bin_col] != df[self.bin_col].shift()).cumsum()).cumcount() + 1
        
        result_columns = [
            'bin_percentile', 'bin_mean', 'bin_std', 'bin_skew',
            'is_extreme_low', 'is_extreme_high', 'is_middle_range',
            'time_in_low', 'time_in_high', 'time_in_middle',
            'bin_up_moves', 'bin_down_moves', 'bin_no_moves',
            'bin_volatility', 'bin_range', 'bin_regime', 'bin_persistence'
        ]
        
        df = self.postprocess(df)
        self.log_calculation_end(df, result_columns)
        
        return df