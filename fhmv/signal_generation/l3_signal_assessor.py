# fhmv/signals/l3_signal_assessor.py

import pandas as pd
import numpy as np

FHMV_REGIME_RHA = "RHA" # Assuming these are accessible or defined in constants.py
FHMV_REGIME_VT = "VT"

class L3SignalAssessor:
    """
    Assesses Liquidation Delta (l3) inflection point signals for non-RC regimes (RHA, VT).
    Incorporates Delta microstructure and contextual feature analysis as per
    Engineering Doc Issue Three and Compendium Section V.B.
    """
    def __init__(self, l3_parameters: dict, non_rc_l3_rules: dict):
        """
        Initializes the L3SignalAssessor.
        (Docstring with example parameters from previous step)
        """
        self.p = l3_parameters
        self.rules = non_rc_l3_rules
        self.min_delta_series_len = max(
            self.p.get('micro_norm_period_delta', 10) + 2,
            self.p.get('micro_slope_window', 2) * 2 + 2, # Ensure enough points for slope calcs
            5 
        )
        # print("L3SignalAssessor Initialized.") # Removed for brevity

    def _detect_l3_inflection(self, delta_series: pd.Series) -> list:
        """Detects l3 inflection points (peaks and troughs) in the Delta series."""
        #
        inflections = []
        if len(delta_series) < 3: return inflections
        
        # Use .iloc for integer-based indexing robustness if delta_series index is not standard RangeIndex
        delta_diff = delta_series.diff().iloc[1:] # First diff is NaN
        if len(delta_diff) < 2: return inflections
        
        delta_diff_sign_change = np.sign(delta_diff).diff().iloc[1:] # First element of this is also NaN
        
        min_abs_delta_change_for_inflection = self.p.get('inflection_min_delta_change_abs', 0.0)

        for t_relative in range(len(delta_diff_sign_change)): # t_relative is index for delta_diff_sign_change
            # Original series index corresponding to this inflection point (where the turn happens)
            # diff() was on delta_series. sign_change was on diff().
            # if delta_diff_sign_change.iloc[t_relative] is at time T, it compares sign(delta_T - delta_{T-1}) with sign(delta_{T-1} - delta_{T-2})
            # So the actual inflection point (the extreme) is delta_series.iloc[t_relative + 1] (index in original diff series)
            # and delta_series.iloc[t_relative + 2] (index in original delta_series)
            # Let's adjust: an inflection is at index `i` if (d[i]-d[i-1]) and (d[i-1]-d[i-2]) have different signs.
            # So, delta_diff_sign_change.index[t_relative] corresponds to the original delta_series index.
            
            original_series_idx_of_inflection_point = delta_diff_sign_change.index[t_relative]
            # This `original_series_idx_of_inflection_point` is where the second of the two diffs was calculated.
            # The actual peak/trough is at this index or the one before.
            # If sign_change at T (based on diff T and diff T-1), peak/trough is at T-1 in original delta series.
            # So, if delta_diff_sign_change is indexed from 0, it corresponds to original index starting from 2.
            # delta_diff_sign_change.index are the original timestamps.
            
            inflection_point_orig_idx = original_series_idx_of_inflection_point
            # To get integer index for delta_series:
            try:
                t_inflection_integer_idx = delta_series.index.get_loc(inflection_point_orig_idx)
            except KeyError:
                continue # Should not happen if indexed correctly

            if t_inflection_integer_idx < 1: continue # Need at least one point before for magnitude check

            # Check magnitude of change around inflection
            # e.g. |delta_series[t_inflection] - delta_series[t_inflection - k]|
            # For simplicity, using change that formed the second part of the turn
            change_magnitude = np.abs(delta_series.iloc[t_inflection_integer_idx] - delta_series.iloc[t_inflection_integer_idx-1])
            if change_magnitude < min_abs_delta_change_for_inflection:
                continue

            if delta_diff_sign_change.iloc[t_relative] == -2.0: # Peak
                inflections.append({'index': inflection_point_orig_idx, 
                                    'time_idx': t_inflection_integer_idx,
                                    'type': 'peak', 
                                    'delta_value': delta_series.loc[inflection_point_orig_idx]})
            elif delta_diff_sign_change.iloc[t_relative] == 2.0: # Trough
                inflections.append({'index': inflection_point_orig_idx,
                                    'time_idx': t_inflection_integer_idx,
                                    'type': 'trough',
                                    'delta_value': delta_series.loc[inflection_point_orig_idx]})
        return inflections[-1:] if inflections else []


    def _calculate_delta_microstructure_features(self, delta_series: pd.Series, inflection_event: dict) -> dict:
        # (Code from previous step, with minor robustness)
        features = {}
        t_inflection = inflection_event['time_idx'] # integer index in delta_series
        norm_period = self.p.get('micro_norm_period_delta', 10)
        if t_inflection >= norm_period and t_inflection > 0:
            recent_delta_sub = delta_series.iloc[max(0, t_inflection - norm_period) : t_inflection]
            recent_delta_std = recent_delta_sub.std()
            raw_magnitude_approx = np.abs(delta_series.iloc[t_inflection] - delta_series.iloc[t_inflection-1])
            if pd.notna(recent_delta_std) and recent_delta_std > self.p.get('epsilon_std', 1e-9):
                features['norm_magnitude_by_delta_std'] = raw_magnitude_approx / recent_delta_std
            else: features['norm_magnitude_by_delta_std'] = 0.0
        else: features['norm_magnitude_by_delta_std'] = 0.0
        
        slope_win = self.p.get('micro_slope_window', 2)
        if t_inflection >= slope_win + 1 and t_inflection < len(delta_series) - slope_win and slope_win > 0:
            slope_before = (delta_series.iloc[t_inflection - 1] - delta_series.iloc[t_inflection - 1 - slope_win]) / slope_win
            slope_after = (delta_series.iloc[t_inflection + slope_win] - delta_series.iloc[t_inflection]) / slope_win
            features['speed_of_turn'] = np.abs(slope_after - slope_before)
        else: features['speed_of_turn'] = 0.0

        if t_inflection >= 2:
            delta_prime_inflection = delta_series.iloc[t_inflection] - delta_series.iloc[t_inflection-1]
            delta_prime_inflection_minus_1 = delta_series.iloc[t_inflection-1] - delta_series.iloc[t_inflection-2]
            features['sharpness_2nd_deriv_approx'] = np.abs(delta_prime_inflection - delta_prime_inflection_minus_1)
        else: features['sharpness_2nd_deriv_approx'] = 0.0
        return features

    def _quantify_contextual_features(self, current_fhmv_state: str,
                                      current_processed_features: pd.Series,
                                      recent_features_history: pd.DataFrame) -> dict:
        # (Code from previous step, with minor robustness for history length)
        context_scores = {}
        current_vol = current_processed_features['Vol']
        vol_thresholds = self.p.get('context_vol_thresholds', {'low': 0.01, 'moderate': 0.03, 'high': 0.05})
        if pd.notna(current_vol):
            if current_vol < vol_thresholds['low']: context_scores['vol_regime_score'] = -1
            elif current_vol < vol_thresholds['moderate']: context_scores['vol_regime_score'] = 0
            elif current_vol < vol_thresholds['high']: context_scores['vol_regime_score'] = 1
            else: context_scores['vol_regime_score'] = 2
        else: context_scores['vol_regime_score'] = 0

        current_abs = current_processed_features['Abs']
        abs_ma_period = self.p.get('context_abs_ma_period', 10)
        abs_trend_period = self.p.get('context_abs_trend_period', 5)
        if len(recent_features_history['Abs']) >= abs_ma_period and pd.notna(current_abs):
            abs_ma = recent_features_history['Abs'].rolling(window=abs_ma_period, min_periods=abs_ma_period//2).mean().iloc[-1]
            if pd.notna(abs_ma): context_scores['abs_level_vs_ma'] = 1 if current_abs > abs_ma else (-1 if current_abs < abs_ma else 0)
            else: context_scores['abs_level_vs_ma'] = 0
            if len(recent_features_history['Abs']) >= abs_trend_period and abs_trend_period > 1:
                abs_slope = recent_features_history['Abs'].diff(abs_trend_period-1).iloc[-1] / (abs_trend_period-1)
                if pd.notna(abs_slope): context_scores['abs_trend_slope'] = np.sign(abs_slope)
                else: context_scores['abs_trend_slope'] = 0
            else: context_scores['abs_trend_slope'] = 0
        else: context_scores['abs_level_vs_ma'] = 0; context_scores['abs_trend_slope'] = 0

        current_rpn = current_processed_features['RPN']
        rpn_extreme_thresh = self.p.get('context_rpn_extreme_thresh', {'low': 0.2, 'high': 0.8})
        if pd.notna(current_rpn):
            context_scores['rpn_level_extreme'] = 1 if current_rpn < rpn_extreme_thresh['low'] or current_rpn > rpn_extreme_thresh['high'] else 0
        else: context_scores['rpn_level_extreme'] = 0
        
        context_scores['lm_sr_level_score'] = 0 
        context_scores['roc_roc2_state_score'] = 0
        return context_scores

    def _aggregate_quality_score(self, micro_features: dict, context_scores: dict, fhmv_state: str) -> tuple[float, str]:
        # (Code from previous step)
        total_score = 0.0; w_micro = self.rules.get('aggregation_weights_micro', {})
        total_score += w_micro.get('norm_magnitude_by_delta_std', 1.0) * micro_features.get('norm_magnitude_by_delta_std',0)
        total_score += w_micro.get('speed_of_turn', 1.0) * micro_features.get('speed_of_turn',0)
        total_score += w_micro.get('sharpness_2nd_deriv_approx', 1.0) * micro_features.get('sharpness_2nd_deriv_approx',0)
        w_context = self.rules.get('aggregation_weights_context', {})
        total_score += w_context.get('vol_regime_score', 1.0) * context_scores.get('vol_regime_score',0)
        total_score += w_context.get('abs_level_vs_ma', 1.0) * context_scores.get('abs_level_vs_ma',0)
        
        conf_thresh = self.rules.get('confidence_thresholds', {"low":0.3, "medium":0.5, "high":0.7})
        confidence = "NONE" # Default if score too low
        if total_score >= conf_thresh['high']: confidence = "HIGH"
        elif total_score >= conf_thresh['medium']: confidence = "MEDIUM"
        elif total_score >= conf_thresh['low']: confidence = "LOW"
        return total_score, confidence
        
    def _apply_rha_vt_hard_filters(self, current_fhmv_state: str, inflection_type: str,
                                   current_processed_features: pd.Series,
                                   recent_features_history: pd.DataFrame,
                                   quality_score: float) -> bool:
        # (Code from previous step)
        if current_fhmv_state == FHMV_REGIME_RHA:
            min_abs_val = self.rules.get('rha_min_abs_value', 100.0) # Example value
            if pd.isna(current_processed_features['Abs']) or current_processed_features['Abs'] < min_abs_val: return False
        elif current_fhmv_state == FHMV_REGIME_VT:
            current_trend_val = current_processed_features['T']; roc_lm = current_processed_features['RoC']
            if pd.isna(current_trend_val) or pd.isna(roc_lm): return False # Not enough info to filter
            is_counter_trend_signal = False
            if inflection_type == 'peak' and current_trend_val > self.p.get('context_trend_positive_thresh', 0): is_counter_trend_signal = True
            elif inflection_type == 'trough' and current_trend_val < self.p.get('context_trend_negative_thresh', 0): is_counter_trend_signal = True
            if is_counter_trend_signal:
                max_roc_for_counter = self.rules.get('vt_roc_lm_counter_max_thresh', 0.05)
                if roc_lm > max_roc_for_counter: return False
        return True

    def assess_l3_signal(self, current_fhmv_state: str, delta_series: pd.Series, 
                           current_processed_features: pd.Series, 
                           recent_features_history: pd.DataFrame) -> dict:
        # (Main orchestration logic from previous step)
        final_signal = {"action": "NONE", "confidence_score": 0.0, "details": {"reason": "No qualifying L3 signal.", "fhmv_state": current_fhmv_state}}
        if current_fhmv_state not in [FHMV_REGIME_RHA, FHMV_REGIME_VT]:
            final_signal["details"]["reason"] = f"L3 assessment not for state {current_fhmv_state}."; return final_signal
        if len(delta_series) < self.min_delta_series_len:
            final_signal["details"]["reason"] = "Insufficient Delta series for L3."; return final_signal

        inflection_events = self._detect_l3_inflection(delta_series)
        if not inflection_events: return final_signal
        last_inflection = inflection_events[-1] 
        
        # Ensure features at inflection are used if inflection is not exactly at current_processed_features.name
        # This requires careful handling of time alignment. For now, using current_processed_features as proxy.
        features_at_inflection_time = current_processed_features # Simplified assumption
        if last_inflection['index'] in recent_features_history.index:
             features_at_inflection_time = recent_features_history.loc[last_inflection['index']]


        micro_features = self._calculate_delta_microstructure_features(delta_series, last_inflection)
        context_scores = self._quantify_contextual_features(current_fhmv_state, features_at_inflection_time, recent_features_history)
        quality_score, confidence_category = self._aggregate_quality_score(micro_features, context_scores, current_fhmv_state)
        
        final_signal["details"].update({
            "raw_quality_score": quality_score, "micro_features": micro_features, 
            "context_scores": context_scores, "inflection_event": last_inflection,
            "confidence_category": confidence_category
        })

        passes_hard_filters = self._apply_rha_vt_hard_filters(current_fhmv_state, last_inflection['type'],
                                                              features_at_inflection_time, recent_features_history, quality_score)
        if not passes_hard_filters:
            final_signal["details"]["reason"] = f"L3 signal failed hard filters for {current_fhmv_state}."; return final_signal

        if quality_score > self.rules.get('min_score_for_signal', 0.1) and confidence_category != "NONE":
            action = "BUY" if last_inflection['type'] == 'trough' else "SELL"
            final_signal["action"] = action
            final_signal["confidence_score"] = quality_score 
            final_signal["details"]["reason"] = f"Qualified L3 signal ({last_inflection['type']}) in {current_fhmv_state}."
        else:
            final_signal["details"]["reason"] = f"L3 signal score {quality_score:.2f} or confidence {confidence_category} below threshold."
        return final_signal