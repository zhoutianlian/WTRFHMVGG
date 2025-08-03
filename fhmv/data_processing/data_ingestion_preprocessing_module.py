# fhmv/data_processing/data_ingestion_preprocessing_module.py

import pandas as pd
import numpy as np

class DataIngestionPreprocessingModule:
    """
    FHMV Data Ingestion and Preprocessing Module.

    This module is responsible for taking raw hourly market data and calculating
    the 10 core FHMV features. It applies necessary boundary condition handling
    and EWMA smoothing for the Delta feature as specified in the FHMV
    Compendium and Engineering Implementation Document.

    The 10 core features are:
    1. LM (Liquidation Magnitude)
    2. SR (Spike Ratio)
    3. Abs (Absorption)
    4. RoC (Rate of Change of Liquidation Magnitude)
    5. RoC2 (Rate of Change of RoC)
    6. P (Price)
    7. T (Trend - ADX)
    8. Vol (Volatility - ATR)
    9. RPN (Liquidation Dominance)
    10. Delta (Net Liquidation Delta, EWMA smoothed)
    """

    def __init__(self, config: dict):
        """
        Initializes the DataIngestionPreprocessingModule with necessary parameters
        from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters like:
                "abs_epsilon": float,
                "abs_lm_threshold_small": float,
                "abs_large_value": float,
                "delta_ewma_alpha": float,
                "vol_atr_period": int,
                "trend_adx_period": int
        """
        self.abs_epsilon = config.get("abs_epsilon", 1e-9)
        self.abs_lm_threshold_small = config.get("abs_lm_threshold_small", 1e-6)
        self.abs_large_value = config.get("abs_large_value", 1e12)
        self.delta_ewma_alpha = config.get("delta_ewma_alpha", 0.3)
        self.vol_atr_period = config.get("vol_atr_period", 14)
        self.trend_adx_period = config.get("trend_adx_period", 14)

        if not (0 < self.delta_ewma_alpha <= 1):
            raise ValueError("Delta EWMA alpha must be between 0 (exclusive) and 1 (inclusive).")
        if self.vol_atr_period <= 0:
            raise ValueError("Volatility ATR period must be positive.")
        if self.trend_adx_period <= 0:
            raise ValueError("Trend ADX period must be positive.")
            
        print("DataIngestionPreprocessingModule Initialized with config.")


    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """
        Calculates Average True Range (ATR).
        ATR is used as the Volatility (Vol) feature.
        """
        if not (isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series)):
            raise TypeError("High, Low, Close must be Pandas Series for ATR calculation.")

        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))

        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
        
        # Standard ATR calculation using EWMA (Wilder's smoothing)
        # alpha = 1 / period for Wilder's smoothing
        atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=max(1, period)).mean() # Ensure min_periods is at least 1
        return atr

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """
        Calculates Average Directional Index (ADX).
        ADX is used as the Trend (T) feature.
        """
        if not (isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series)):
            raise TypeError("High, Low, Close must be Pandas Series for ADX calculation.")

        move_up = high.diff()
        move_down = low.shift(1) - low # Corrected: low.diff() is low_t - low_{t-1}, so prev_low - low_t is low.shift(1) - low

        plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=high.index)

        # Smoothed +DM, -DM using Wilder's smoothing (EWMA with alpha = 1/period)
        alpha_wilder = 1.0 / period
        
        # True Range (TR) is needed for normalization in DI calculation
        high_low_diff = high - low
        high_prev_close_abs = np.abs(high - close.shift(1))
        low_prev_close_abs = np.abs(low - close.shift(1))
        tr = pd.concat([high_low_diff, high_prev_close_abs, low_prev_close_abs], axis=1).max(axis=1, skipna=False)
        atr_for_adx = tr.ewm(alpha=alpha_wilder, adjust=False, min_periods=max(1,period)).mean()
        
        atr_for_adx_safe = atr_for_adx.replace(0, np.nan) # Avoid division by zero

        smoothed_plus_dm = plus_dm.ewm(alpha=alpha_wilder, adjust=False, min_periods=max(1,period)).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=alpha_wilder, adjust=False, min_periods=max(1,period)).mean()

        plus_di = (smoothed_plus_dm / atr_for_adx_safe) * 100
        minus_di = (smoothed_minus_dm / atr_for_adx_safe) * 100
        
        plus_di = plus_di.fillna(0) 
        minus_di = minus_di.fillna(0)

        dx_abs_diff = np.abs(plus_di - minus_di)
        dx_sum = plus_di + minus_di
        dx_sum_safe = dx_sum.replace(0, np.nan) 
        
        dx = (dx_abs_diff / dx_sum_safe) * 100
        dx = dx.fillna(0) 

        adx = dx.ewm(alpha=alpha_wilder, adjust=False, min_periods=max(1,period)).mean()
        return adx

    def process_raw_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw hourly market data to calculate the 10 core FHMV features.
       

        Args:
            raw_df (pd.DataFrame): DataFrame containing raw market data. Expected columns:
                'timestamp' (should be index or convertible to index)
                'high' (float): High prices for the hour.
                'low' (float): Low prices for the hour.
                'close' (float): Closing prices for the hour.
                'LM' (float): Liquidation Magnitude.
                'SR' (float): Spike Ratio.
                'rpn_long_liqs' (float): Volume of long liquidations for RPN.
                'rpn_short_liqs' (float): Volume of short liquidations for RPN.
                'delta_long_liq_vol' (float): Volume of long liquidations for Delta.
                'delta_short_liq_vol' (float): Volume of short liquidations for Delta.

        Returns:
            pd.DataFrame: DataFrame with a timestamp index and the 10 core FHMV features.
        """
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a Pandas DataFrame.")

        if 'timestamp' in raw_df.columns:
            df = raw_df.set_index('timestamp').copy() # Use copy to avoid SettingWithCopyWarning
        else:
            df = raw_df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"DataFrame index could not be converted to DatetimeIndex: {e}")

        core_features = pd.DataFrame(index=df.index)

        # Features 1 & 2: LM, SR (Direct input) [cite: 20]
        core_features['LM'] = df['LM']
        core_features['SR'] = df['SR']

        # Feature 6: P (Price) - Using 'close' price [cite: 20]
        core_features['P'] = df['close']
        
        price_change_abs_val = np.abs(core_features['P'].diff())

        # Feature 3: Abs (Absorption)
        abs_val = pd.Series(index=df.index, dtype=float)
        valid_delta_p_mask = (price_change_abs_val >= self.abs_epsilon) & (price_change_abs_val.notna())
        abs_val.loc[valid_delta_p_mask] = core_features['LM'].loc[valid_delta_p_mask] / price_change_abs_val.loc[valid_delta_p_mask]
        
        boundary_cond_2_mask = (price_change_abs_val < self.abs_epsilon) & (core_features['LM'] > self.abs_lm_threshold_small) & (price_change_abs_val.notna()) & (core_features['LM'].notna())
        abs_val.loc[boundary_cond_2_mask] = self.abs_large_value
        
        boundary_cond_3_mask = (core_features['LM'] <= self.abs_lm_threshold_small) & (price_change_abs_val < self.abs_epsilon) & (core_features['LM'].notna()) & (price_change_abs_val.notna())
        abs_val.loc[boundary_cond_3_mask] = np.nan 
        
        abs_val.replace([np.inf, -np.inf], self.abs_large_value, inplace=True)
        core_features['Abs'] = abs_val

        # Feature 4: RoC (Rate of Change of Liquidation Magnitude) [cite: 20]
        core_features['RoC'] = core_features['LM'].diff()

        # Feature 5: RoC2 (Rate of Change of RoC) [cite: 20]
        core_features['RoC2'] = core_features['RoC'].diff()

        # Feature 8: Vol (Volatility) - Using ATR [cite: 20]
        core_features['Vol'] = self._calculate_atr(df['high'], df['low'], df['close'], self.vol_atr_period)

        # Feature 7: T (Trend) - Using ADX [cite: 20]
        core_features['T'] = self._calculate_adx(df['high'], df['low'], df['close'], self.trend_adx_period)

        # Feature 9: RPN (Liquidation Dominance) [cite: 20, 25, 1051]
        # Corrected formula: LL / (LL + SL)
        long_liqs_rpn = df['rpn_long_liqs']
        short_liqs_rpn = df['rpn_short_liqs']
        total_liqs_rpn = long_liqs_rpn + short_liqs_rpn
        
        rpn_val = pd.Series(index=df.index, dtype=float)
        valid_total_liqs_mask = (total_liqs_rpn != 0) & total_liqs_rpn.notna()
        rpn_val.loc[valid_total_liqs_mask] = long_liqs_rpn.loc[valid_total_liqs_mask] / total_liqs_rpn.loc[valid_total_liqs_mask]
        
        zero_total_liqs_mask = (total_liqs_rpn == 0) | total_liqs_rpn.isna() # Handle NaN as if zero for boundary
        rpn_val.loc[zero_total_liqs_mask] = 0.5
        core_features['RPN'] = rpn_val

        # Feature 10: Delta (Net Liquidation Delta, EWMA smoothed)
        long_liq_vol_delta = df['delta_long_liq_vol']
        short_liq_vol_delta = df['delta_short_liq_vol']
        total_liq_vol_delta_constituents = long_liq_vol_delta + short_liq_vol_delta
        
        raw_delta = long_liq_vol_delta - short_liq_vol_delta
        raw_delta.loc[total_liq_vol_delta_constituents == 0] = 0.0
        raw_delta.loc[total_liq_vol_delta_constituents.isna()] = 0.0 # If constituents are NaN, treat as no liq for Delta

        if raw_delta.empty:
            core_features['Delta'] = pd.Series(dtype=float, index=df.index)
        else:
            # Ensure raw_delta is float before ewm, handle potential all-NaN after boundary conditions
            if raw_delta.isnull().all():
                 core_features['Delta'] = raw_delta # Propagate all NaNs
            else:
                 # Using adjust=True for standard EWMA behavior for finite series.
                 # min_periods=1 to start EWMA as soon as first non-NaN value appears.
                 core_features['Delta'] = raw_delta.ewm(alpha=self.delta_ewma_alpha, adjust=True, min_periods=1).mean()
        
        return core_features[[
            'LM', 'SR', 'Abs', 'RoC', 'RoC2', 'P', 'T', 'Vol', 'RPN', 'Delta'
        ]]