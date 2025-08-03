# fhmv/signals/clad_signal_generator.py

import pandas as pd
import numpy as np

# Assuming these constants are defined or passed if names are used instead of indices/booleans
UPPER_BOUNDARY = "UPPER"
LOWER_BOUNDARY = "LOWER"

class CLADSignalGenerator:
    """
    Implements the Critical Liquidation-Absorption Disequilibrium (CLAD) principle
    for identifying high-probability reversal points at RC boundaries.
    Based on Compendium Section III.B, V.A and Eng. Doc. Issue Six / Compendium VII.B.
   
    """
    def __init__(self, clad_parameters: dict):
        """
        Initializes the CLADSignalGenerator with necessary parameters.
        (Docstring with example clad_parameters from previous step)
        """
        self.params = clad_parameters
        self.history_lookback = self.params.get("history_lookback_for_stats", 50)
        if self.history_lookback <= 0 : self.history_lookback = 1 # Defensive
        # print("CLADSignalGenerator Initialized.") # Removed for brevity in combined output

    def _module1_is_rc_environment(self, current_vol: float, current_trend_strength: float) -> bool:
        """CLAD Module 1: Is the current context range-bound oscillation? (CLAD Condition 4)"""
        #
        is_trend_weak = current_trend_strength < self.params.get("module1_trend_strength_thresh", 25.0)
        is_vol_in_rc_range = (self.params.get("module1_rc_vol_lower_thresh", 0.0) < current_vol <
                                self.params.get("module1_rc_vol_upper_thresh", np.inf))
        return is_trend_weak and is_vol_in_rc_range

    def _module2_is_boundary_genuinely_tested(self, current_price: float, current_lm: float,
                                             current_delta: float, current_sr: float,
                                             dynamic_rc_boundaries: dict,
                                             historical_features: pd.DataFrame) -> tuple[bool, str | None]:
        """CLAD Module 2: Is the boundary being genuinely tested? (CLAD Condition 1)"""
        #
        upper_b = dynamic_rc_boundaries.get("upper", np.nan)
        lower_b = dynamic_rc_boundaries.get("lower", np.nan)

        if np.isnan(upper_b) or np.isnan(lower_b) or upper_b <= lower_b + 1e-9: # Added tolerance for upper_b > lower_b
            return False, None

        price_boundary_proximity_factor = self.params.get("module2_price_boundary_proximity_factor", 0.001) # e.g., 0.1%
        price_near_upper = (upper_b - current_price) <= (price_boundary_proximity_factor * upper_b) if upper_b != 0 else False
        price_near_lower = (current_price - lower_b) <= (price_boundary_proximity_factor * lower_b) if lower_b != 0 else False
        
        tested_boundary = None
        if price_near_upper: tested_boundary = UPPER_BOUNDARY
        elif price_near_lower: tested_boundary = LOWER_BOUNDARY
        if tested_boundary is None: return False, None

        recent_lm = historical_features['LM'].tail(self.history_lookback).dropna()
        if recent_lm.empty: return False, None # Not enough history
        mean_lm_rc = recent_lm.mean()
        std_lm_rc = max(recent_lm.std(), 1e-9) # Avoid zero std
        is_lm_significant = current_lm > (mean_lm_rc + self.params.get("module2_lm_std_dev_k1", 1.0) * std_lm_rc)
        
        is_sr_high = current_sr > self.params.get("module2_sr_significance_thresh", 1.5)
        lm_condition_met = is_lm_significant # Simplified from (is_lm_significant or (is_lm_significant and is_sr_high))
        if self.params.get("module2_use_sr_enhancement", False) and is_sr_high: # Optional SR check
            lm_condition_met = lm_condition_met or is_lm_significant # If SR is used, it enhances, not replaces LM check

        if not lm_condition_met: return False, None
            
        recent_abs_delta = np.abs(historical_features['Delta'].tail(self.history_lookback).dropna())
        if recent_abs_delta.empty: return False, None
        mean_abs_delta_rc = recent_abs_delta.mean()
        std_abs_delta_rc = max(recent_abs_delta.std(), 1e-9)
        is_delta_magnitude_significant = np.abs(current_delta) > (mean_abs_delta_rc + self.params.get("module2_delta_std_dev_k2", 1.0) * std_abs_delta_rc)

        delta_directionally_consistent = False
        if tested_boundary == UPPER_BOUNDARY and current_delta > self.params.get("module2_delta_dir_thresh", 0): 
            delta_directionally_consistent = True
        elif tested_boundary == LOWER_BOUNDARY and current_delta < -self.params.get("module2_delta_dir_thresh", 0):
            delta_directionally_consistent = True
        
        return is_delta_magnitude_significant and delta_directionally_consistent, tested_boundary


    def _module3_is_liquidation_attack_impeded(self, current_lm: float, current_abs_feature: float,
                                             roc_flow: float, roc2_flow: float, 
                                             historical_features: pd.DataFrame) -> bool:
        """CLAD Module 3: Has the liquidation attack failed? (CLAD Condition 2)"""
        #
        recent_abs_feat = historical_features['Abs'].tail(self.history_lookback).dropna()
        if recent_abs_feat.empty: absorption_overwhelming = False # Default if no history
        else:
            mean_abs_rc = recent_abs_feat.mean()
            std_abs_rc = max(recent_abs_feat.std(), 1e-9)
            is_abs_statistically_high = current_abs_feature > (mean_abs_rc + self.params.get("module3_abs_std_dev_k3", 1.5) * std_abs_rc)
            is_abs_conceptually_infinite = current_abs_feature >= self.params.get("module3_abs_large_value_on_zero_delta_p", 1e12) * 0.99
            absorption_overwhelming = is_abs_statistically_high or is_abs_conceptually_infinite
        
        is_roc_exhausted = roc_flow < self.params.get("module3_roc_flow_exhaust_thresh", 0.05)
        is_roc2_significantly_negative = roc2_flow < self.params.get("module3_roc2_flow_sig_neg_thresh", -0.01)
        exhaustion_met = is_roc_exhausted or is_roc2_significantly_negative
        
        return absorption_overwhelming or exhaustion_met

    def _module4_is_net_pressure_turn_confirmed(self, recent_delta_series: pd.Series,
                                               recent_rpn_series: pd.Series,
                                               tested_boundary_direction: str) -> bool:
        """CLAD Module 4: Has net liquidation pressure confirmed a turn? (CLAD Condition 3)"""
        #
        min_len_for_ma = max(self.params.get("module4_rpn_ma_long_period",5), 
                             self.params.get("module4_delta_roc_periods", 1) + 1, 2)
        if len(recent_delta_series) < min_len_for_ma or len(recent_rpn_series) < min_len_for_ma:
            return False

        delta_roc_periods = self.params.get("module4_delta_roc_periods", 1)
        delta_roc = recent_delta_series.diff(delta_roc_periods).iloc[-1] if len(recent_delta_series) > delta_roc_periods else np.nan
        
        delta_inflection_confirms_turn = False
        if pd.notna(delta_roc):
            if tested_boundary_direction == UPPER_BOUNDARY and delta_roc < -self.params.get("module4_delta_roc_abs_thresh", 0): 
                delta_inflection_confirms_turn = True
            elif tested_boundary_direction == LOWER_BOUNDARY and delta_roc > self.params.get("module4_delta_roc_abs_thresh", 0):
                delta_inflection_confirms_turn = True
        
        rpn_confirms_turn = False
        current_rpn = recent_rpn_series.iloc[-1]
        rpn_ma_short_period = self.params.get("module4_rpn_ma_short_period",3)
        rpn_ma_long_period = self.params.get("module4_rpn_ma_long_period",5)

        if len(recent_rpn_series) >= rpn_ma_long_period:
            rpn_ma_short = recent_rpn_series.rolling(window=rpn_ma_short_period).mean().iloc[-1]
            rpn_ma_long = recent_rpn_series.rolling(window=rpn_ma_long_period).mean().iloc[-1]
            rpn_roc_val = recent_rpn_series.diff().iloc[-1] if len(recent_rpn_series) >= 2 else 0.0

            rpn_roc_strong_thresh = self.params.get("module4_rpn_roc_thresh_strong", 0.05)

            if tested_boundary_direction == UPPER_BOUNDARY:
                rpn_was_low = recent_rpn_series.iloc[-rpn_ma_short_period:-1].mean() < self.params.get("module4_rpn_recent_low_thresh", 0.4) if len(recent_rpn_series) >= rpn_ma_short_period+1 else True
                is_rpn_trending_up = (current_rpn > rpn_ma_short and rpn_ma_short > rpn_ma_long and rpn_was_low) or \
                                     (rpn_roc_val > rpn_roc_strong_thresh and rpn_was_low)
                if is_rpn_trending_up: rpn_confirms_turn = True
            elif tested_boundary_direction == LOWER_BOUNDARY:
                rpn_was_high = recent_rpn_series.iloc[-rpn_ma_short_period:-1].mean() > self.params.get("module4_rpn_recent_high_thresh", 0.6) if len(recent_rpn_series) >= rpn_ma_short_period+1 else True
                is_rpn_trending_down = (current_rpn < rpn_ma_short and rpn_ma_short < rpn_ma_long and rpn_was_high) or \
                                       (rpn_roc_val < -rpn_roc_strong_thresh and rpn_was_high)
                if is_rpn_trending_down: rpn_confirms_turn = True
        
        return delta_inflection_confirms_turn or rpn_confirms_turn # Compendium: AND/OR, stronger if both. Using OR for now.

    def generate_clad_signal(self, current_processed_features: pd.Series,
                               recent_features_history: pd.DataFrame, 
                               dynamic_rc_boundaries: dict) -> dict:
        # (Orchestration logic from previous step, minor cleanups)
        min_hist_len = max(self.history_lookback, 
                           self.params.get("module4_rpn_ma_long_period",5) +1, # +1 for diff
                           2) 
        if len(recent_features_history) < min_hist_len:
            return {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "Insufficient history for CLAD."}}

        module1_pass = self._module1_is_rc_environment(
            current_vol=current_processed_features['Vol'],
            current_trend_strength=current_processed_features['T']
        )
        if not module1_pass:
            return {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "CLAD M1 Failed: Not RC Env."}}

        # Use history *before* current point for M2 stats, but current values for LM, Delta etc.
        hist_for_m2_stats = recent_features_history.iloc[:-1] if len(recent_features_history) > 1 else recent_features_history
        if hist_for_m2_stats.empty and self.history_lookback > 0 : # Need history for stats
             return {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "CLAD M2 Failed: Not enough history for stats."}}


        module2_pass, tested_boundary = self._module2_is_boundary_genuinely_tested(
            current_price=current_processed_features['P'], current_lm=current_processed_features['LM'],
            current_delta=current_processed_features['Delta'], current_sr=current_processed_features['SR'],
            dynamic_rc_boundaries=dynamic_rc_boundaries, historical_features=hist_for_m2_stats
        )
        if not module2_pass:
            return {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "CLAD M2 Failed: Boundary not genuinely tested."}}

        module3_pass = self._module3_is_liquidation_attack_impeded(
            current_lm=current_processed_features['LM'], current_abs_feature=current_processed_features['Abs'],
            roc_flow=current_processed_features['RoC'], roc2_flow=current_processed_features['RoC2'],
            historical_features=hist_for_m2_stats
        )
        if not module3_pass:
            return {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "CLAD M3 Failed: Attack not impeded."}}

        module4_pass = self._module4_is_net_pressure_turn_confirmed(
            recent_delta_series=recent_features_history['Delta'], # Includes current point for RoC calc
            recent_rpn_series=recent_features_history['RPN'],     # Includes current point
            tested_boundary_direction=tested_boundary
        )
        if not module4_pass:
            return {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "CLAD M4 Failed: Turn not confirmed."}}

        action = "SELL" if tested_boundary == UPPER_BOUNDARY else ("BUY" if tested_boundary == LOWER_BOUNDARY else "NONE")
        confidence = self.params.get("clad_signal_base_confidence", 0.75)
        
        return {"action": action, "confidence_score": confidence,
                "details": {"reason": f"CLAD signal: {action} at {tested_boundary} boundary.",
                            "tested_boundary": tested_boundary, "clad_modules_passed": [1,2,3,4]}}