# fhmv/risk_management/hpem_subtype_assessor.py

import pandas as pd
import numpy as np

class HPEMSubTypeAssessor:
    """
    Assesses HPEM Sub-Types: Bullish/Bearish HPEM and Short Squeeze Possibility.
    Implements logic from Engineering Doc Issue Four.
    This is a placeholder implementation and needs full logic as per the document.
    """
    def __init__(self, hpem_subtype_params: dict):
        """
        Initializes the HPEMSubTypeAssessor.

        Args:
            hpem_subtype_params (dict): Parameters for HPEM subtype differentiation
                                        and squeeze assessment. Example keys:
                                        'price_strong_positive_thresh', 'T_strong_positive_thresh',
                                        'RPN_bullish_HPEM_thresh', 'Delta_bullish_HPEM_thresh',
                                        'decision_min_score_thresh', 'score_diff_margin',
                                        'RPN_squeeze_extreme_thresh', 'LM_squeeze_extreme_thresh_val',
                                        'SR_squeeze_extreme_thresh_val', 'RoC2_squeeze_accelerating_thresh',
                                        'Abs_HPEM_low_thresh',
                                        'squeeze_decision_high_confidence_thresh',
                                        'squeeze_decision_medium_confidence_thresh'
        """
        self.p = hpem_subtype_params
        # print("HPEMSubTypeAssessor Initialized.") # Removed for brevity

    def _differentiate_hpem_direction(self, current_features: pd.Series, 
                                      price_change: float) -> str:
        """
        Differentiates Bullish vs. Bearish HPEM.
        Based on Eng Doc Issue Four, Section 1.
        """
        score_bullish = 0
        score_bearish = 0

        # Price Change Score
        if price_change > self.p.get('price_strong_positive_thresh', 0.002): score_bullish += 2
        elif price_change > self.p.get('price_moderate_positive_thresh', 0.001): score_bullish += 1
        if price_change < -self.p.get('price_strong_positive_thresh', 0.002): score_bearish += 2 # Assuming symmetrical thresholds
        elif price_change < -self.p.get('price_moderate_positive_thresh', 0.001): score_bearish += 1
        
        # Trend (T) Score
        t_val = current_features.get('T', 0)
        if t_val > self.p.get('T_strong_positive_thresh', 30): score_bullish += 2 # ADX example
        elif t_val > self.p.get('T_moderate_positive_thresh', 20): score_bullish += 1
        if t_val < self.p.get('T_strong_negative_thresh', -30): score_bearish += 2 # Assuming T can be negative for ADX-like
        elif t_val < self.p.get('T_moderate_negative_thresh', -20): score_bearish += 1

        # RPN Score (Pro-Trend Liquidation Bias)
        # RPN = LL / (LL+SL). For Bullish HPEM (shorts liquidated), RPN should be low.
        # Eng Doc Issue 4 (Table 4 & text [cite: 901, 946]) says Bullish HPEM has RPN > 0.7 (shorts liq).
        # This implies RPN definition might be SL / (SL+LL) in that context, or my RPN interpretation for HPEM is reversed.
        # Let's stick to RPN = LL/(LL+SL) and adjust logic for HPEM direction:
        # Bullish HPEM (upward price, shorts covering/forced out): Short Liqs high, Long Liqs low => RPN LOW.
        # Bearish HPEM (downward price, longs forced out): Long Liqs high, Short Liqs low => RPN HIGH.
        # This seems to contradict table 1 in Compendium (RPN "Extreme Bias (≈0 or ≈1, Pro-Trend)").
        # Re-checking Compendium III.A Table for HPEM RPN: "Extreme Bias (≈0 or ≈1, Pro-Trend)".
        # If Pro-Trend: Bullish HPEM (trend up) means RPN near 1 (Longs liq dominating? No, that's contra).
        # Let's assume RPN "Pro-Trend" means:
        # - Bullish HPEM (price up): RPN should indicate short liquidations are dominant. If RPN = LL/(LL+SL), this means RPN is LOW (near 0).
        # - Bearish HPEM (price down): RPN should indicate long liquidations are dominant. If RPN = LL/(LL+SL), this means RPN is HIGH (near 1).
        # The Eng Doc Issue 4 logic with "RPN_bullish_HPEM_thresh > 0.7" for bullish implies their RPN definition there was ShortLiqs/(TotalLiqs).
        # Sticking to my current consistent RPN = LL/(LL+SL):
        rpn_val = current_features.get('RPN', 0.5)
        if rpn_val < self.p.get('RPN_bullish_HPEM_thresh_low', 0.3): # Shorts liq dominant for Bullish
            score_bullish += 2
        if rpn_val > self.p.get('RPN_bearish_HPEM_thresh_high', 0.7): # Longs liq dominant for Bearish
            score_bearish += 2

        # Delta Score (Pro-Trend Net Liquidation Flow)
        delta_val = current_features.get('Delta', 0)
        if delta_val > self.p.get('Delta_bullish_HPEM_thresh_pos', 0.1): # Positive Delta for Bullish
             score_bullish += 1
        if delta_val < self.p.get('Delta_bearish_HPEM_thresh_neg', -0.1): # Negative Delta for Bearish
             score_bearish += 1

        # Decision Logic
        if score_bullish >= self.p.get('decision_min_score_thresh', 3) and \
           score_bullish > (score_bearish + self.p.get('score_diff_margin', 1)):
            return "BULLISH"
        elif score_bearish >= self.p.get('decision_min_score_thresh', 3) and \
             score_bearish > (score_bullish + self.p.get('score_diff_margin', 1)):
            return "BEARISH"
        else:
            return "UNDETERMINED"


    def _assess_short_squeeze_possibility(self, current_features: pd.Series, is_bullish_hpem: bool) -> str:
        """
        Assesses Short Squeeze Possibility within a Bullish HPEM context.
        Based on Eng Doc Issue Four, Section 2.
        RPN = LL/(LL+SL). For short squeeze (shorts liq), RPN should be low.
        """
        if not is_bullish_hpem:
            return "NOT_APPLICABLE"

        squeeze_score = 0
        rpn_val = current_features.get('RPN', 0.5)
        lm_val = current_features.get('LM', 0)
        sr_val = current_features.get('SR', 0)
        roc2_lm_val = current_features.get('RoC2', 0) # Assuming RoC2 is RoC2_LM
        abs_val = current_features.get('Abs', np.inf)

        # RPN Extremity (Shorts dominating liquidations -> RPN LOW) [cite: 918, 946]
        if rpn_val < self.p.get('RPN_squeeze_extreme_thresh', 0.05): squeeze_score += 3
        elif rpn_val < self.p.get('RPN_squeeze_very_low_thresh', 0.10): squeeze_score += 2
        elif rpn_val < self.p.get('RPN_bullish_HPEM_thresh_low', 0.35): squeeze_score +=1 # Consistent with Bullish HPEM
        
        # Liquidation Scale & Spike [cite: 919, 946]
        # These thresholds should ideally be percentile-based from config. Using placeholder values.
        if lm_val > self.p.get('LM_squeeze_extreme_thresh_val', 1000): squeeze_score += 2
        if sr_val > self.p.get('SR_squeeze_extreme_thresh_val', 5.0): squeeze_score += 2
        
        # Liquidation Acceleration (RoC2_LM) [cite: 919, 946]
        if roc2_lm_val > self.p.get('RoC2_squeeze_accelerating_thresh', 0.2): squeeze_score += 2
        
        # Absorption Confirmation (Low Abs for HPEM) [cite: 915, 919, 946]
        if abs_val < self.p.get('Abs_HPEM_low_thresh', 50.0): squeeze_score += 1
            
        # Optional: Context of prior bearish/neutral trend (not implemented here for brevity) [cite: 920]

        # Final Possibility Assessment
        if squeeze_score >= self.p.get('squeeze_decision_high_confidence_thresh', 7): return "HIGH"
        elif squeeze_score >= self.p.get('squeeze_decision_medium_confidence_thresh', 5): return "MEDIUM"
        else: return "LOW"

    def assess_hpem_subtype(self, current_features: pd.Series, 
                              previous_price: float) -> dict:
        """
        Main assessment method for HPEM subtype.
        """
        price_change = current_features['P'] - previous_price
        
        direction = self._differentiate_hpem_direction(current_features, price_change)
        
        squeeze_possibility = "NOT_APPLICABLE"
        if direction == "BULLISH":
            squeeze_possibility = self._assess_short_squeeze_possibility(current_features, is_bullish_hpem=True)
            
        return {
            "direction": direction, # "BULLISH", "BEARISH", "UNDETERMINED"
            "squeeze_possibility": squeeze_possibility, # "HIGH", "MEDIUM", "LOW", "NOT_APPLICABLE"
            "details": {
                "price_change_for_direction": price_change
            }
        }