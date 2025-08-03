# fhmv/risk_management/risk_management_module.py

import pandas as pd
import numpy as np
from .hpem_subtype_assessor import HPEMSubTypeAssessor # Relative import

# Define FHMV Regime constants if not imported from a central place
FHMV_REGIME_ST = "ST"; FHMV_REGIME_VT = "VT"; FHMV_REGIME_RC = "RC"
FHMV_REGIME_RHA = "RHA"; FHMV_REGIME_HPEM = "HPEM"; FHMV_REGIME_AMB = "AMB"

class RiskManagementModule:
    """
    FHMV Risk Management Module.
    (Full docstring from previous step)
    """

    def __init__(self, config: dict):
        """
        Initializes the RiskManagementModule.
        Args:
            config (dict): Configuration dictionary containing:
                           "risk_management_config": dict for general risk rules,
                           "hpem_subtype_config": dict for HPEMSubTypeAssessor,
                           "portfolio_config": dict for capital and risk per trade.
        """
        self.params = config.get("risk_management_config", {})
        hpem_subtype_cfg = config.get("hpem_subtype_config", {})
        self.portfolio_cfg = config.get("portfolio_config", {})
        
        self.hpem_assessor = HPEMSubTypeAssessor(hpem_subtype_cfg)
        
        self.regime_names_map_idx_to_str = { # Example, could be part of utils or config
            0: FHMV_REGIME_ST, 1: FHMV_REGIME_VT, 2: FHMV_REGIME_RC,
            3: FHMV_REGIME_RHA, 4: FHMV_REGIME_HPEM, 5: FHMV_REGIME_AMB
        }
        # print("RiskManagementModule Initialized.") # Removed for brevity

    def _apply_amb_protocol(self, raw_signal: dict, current_portfolio: dict) -> tuple[dict, bool]:
        # (Code from previous step)
        new_trade_suppressed = True 
        adjusted_signal = raw_signal.copy()
        adjusted_signal["signal_action"] = "HOLD" 
        adjusted_signal["details"]["risk_reason"] = "AMB Protocol: New trades suppressed."
        adjusted_signal["risk_adjusted_quantity"] = 0
        return adjusted_signal, new_trade_suppressed

    def _calculate_position_size(self, fhmv_state_name: str, signal_confidence: float,
                                 current_price: float, current_vol_atr: float,
                                 hpem_assessment: dict = None) -> float:
        # (Code from previous step, minor refinement for clarity)
        base_size_factor = self.params.get('base_position_size_factor', 1.0)
        conf_min, conf_max = self.params.get('confidence_scaling_factor_min_max', [0.5, 1.2])
        confidence_scaler = conf_min + (conf_max - conf_min) * float(signal_confidence)

        state_multipliers = self.params.get('state_risk_multipliers', {})
        
        # Construct specific key for HPEM if applicable
        hpem_key_suffix = ""
        if fhmv_state_name == FHMV_REGIME_HPEM and hpem_assessment:
            if hpem_assessment["direction"] == "BULLISH" and hpem_assessment["squeeze_possibility"] == "HIGH":
                hpem_key_suffix = "_BULLISH_SQUEEZE_HIGH"
            elif hpem_assessment["direction"] == "BULLISH":
                 hpem_key_suffix = "_BULLISH" # General bullish
            elif hpem_assessment["direction"] == "BEARISH":
                hpem_key_suffix = "_BEARISH"
        
        state_key_for_multiplier = fhmv_state_name + hpem_key_suffix if hpem_key_suffix else fhmv_state_name
        state_scaler = state_multipliers.get(state_key_for_multiplier, state_multipliers.get(fhmv_state_name, 0.5))


        calculated_size_factor = base_size_factor * confidence_scaler * state_scaler

        # Monetary Sizing (simplified example, actual quantity calculation)
        # Placeholder: for now, calculated_size_factor is the output quantity directly.
        # A full implementation would use portfolio_cfg (capital, risk_per_trade)
        # and stop_loss_distance to determine actual units.
        # E.g., stop_loss_distance_abs = atr_stop_mult * current_vol_atr
        # risk_amount = self.portfolio_cfg.get('total_capital', 1e5) * self.portfolio_cfg.get('max_account_risk_per_trade', 0.01)
        # units = risk_amount / stop_loss_distance_abs if stop_loss_distance_abs > 1e-5 else 0
        # return max(0, units * calculated_size_factor) # if calculated_size_factor was meant to scale this base unit size

        return max(0, calculated_size_factor) # Returning the factor as quantity for now


    def _determine_stop_loss_take_profit(self, raw_signal_action: str, fhmv_state_name: str,
                                         entry_price: float, current_vol_atr: float,
                                         dynamic_rc_boundaries: dict = None,
                                         signal_details: dict = None) -> tuple[float | None, float | None]:
        # (Code from previous step, uses signal_details for CLAD tested_boundary)
        sl_price, tp_price = None, None
        
        sl_atr_multipliers = self.params.get('stop_loss_atr_multiplier', {})
        tp_atr_multipliers = self.params.get('take_profit_atr_multiplier', {})
        
        sl_atr_multiplier = sl_atr_multipliers.get(fhmv_state_name, sl_atr_multipliers.get("DEFAULT", 2.0))
        tp_atr_multiplier = tp_atr_multipliers.get(fhmv_state_name, tp_atr_multipliers.get("DEFAULT", 3.0))

        if pd.isna(current_vol_atr) or current_vol_atr <= 1e-9: # Cannot use ATR based SL/TP
            return None, None

        if raw_signal_action == "BUY":
            sl_price = entry_price - sl_atr_multiplier * current_vol_atr
            tp_price = entry_price + tp_atr_multiplier * current_vol_atr
            if fhmv_state_name == FHMV_REGIME_RC and dynamic_rc_boundaries and pd.notna(dynamic_rc_boundaries.get("upper")):
                tp_price = dynamic_rc_boundaries["upper"]
                # For RC BUY, SL might be below the tested LOWER boundary
                tested_boundary = signal_details.get("tested_boundary") if signal_details else None
                if tested_boundary == "LOWER" and pd.notna(dynamic_rc_boundaries.get("lower")):
                     sl_price = dynamic_rc_boundaries.get("lower") - self.params.get("rc_sl_boundary_offset_atr", 0.5) * current_vol_atr
        elif raw_signal_action == "SELL":
            sl_price = entry_price + sl_atr_multiplier * current_vol_atr
            tp_price = entry_price - tp_atr_multiplier * current_vol_atr
            if fhmv_state_name == FHMV_REGIME_RC and dynamic_rc_boundaries and pd.notna(dynamic_rc_boundaries.get("lower")):
                tp_price = dynamic_rc_boundaries["lower"]
                tested_boundary = signal_details.get("tested_boundary") if signal_details else None
                if tested_boundary == "UPPER" and pd.notna(dynamic_rc_boundaries.get("upper")):
                     sl_price = dynamic_rc_boundaries.get("upper") + self.params.get("rc_sl_boundary_offset_atr", 0.5) * current_vol_atr
        return sl_price, tp_price


    def manage_trade_signal(self, raw_signal: dict, current_fhmv_state_idx: int,
                              current_processed_features: pd.Series,
                              recent_features_history: pd.DataFrame, # For P_prev in HPEM, RC boundaries context
                              current_portfolio_status: dict) -> dict:
        # (Main orchestration logic from previous step, with P_prev for HPEM)
        final_trade_decision = raw_signal.copy()
        final_trade_decision.update({
            "risk_adjusted_quantity": 0.0, "stop_loss_price": None,
            "take_profit_price": None, "risk_reason": "No action after risk assessment.",
            "trade_action": "NO_ACTION" # Default for execution module
        })
        fhmv_state_name = self.regime_names_map_idx_to_str.get(current_fhmv_state_idx, FHMV_REGIME_AMB)

        if fhmv_state_name == FHMV_REGIME_AMB:
            adjusted_signal, suppress_new = self._apply_amb_protocol(raw_signal, current_portfolio_status)
            if suppress_new and raw_signal["signal_action"] not in ["HOLD", "NONE"]:
                return adjusted_signal 
        
        if raw_signal["signal_action"] in ["HOLD", "NONE"]:
            final_trade_decision["risk_reason"] = "Initial signal was HOLD/NONE."; return final_trade_decision

        hpem_assessment_result = None
        if fhmv_state_name == FHMV_REGIME_HPEM:
            previous_price = recent_features_history['P'].iloc[-2] if len(recent_features_history['P']) >= 2 else current_processed_features['P']
            hpem_assessment_result = self.hpem_assessor.assess_hpem_subtype(current_processed_features, previous_price)
            final_trade_decision["details"]["hpem_assessment"] = hpem_assessment_result
        
        entry_price = current_processed_features['P']
        current_vol_atr = current_processed_features['Vol']
        
        quantity = self._calculate_position_size(
            fhmv_state_name, raw_signal.get("confidence", 0.0), # Use 0 if confidence missing
            entry_price, current_vol_atr, hpem_assessment_result)
        final_trade_decision["risk_adjusted_quantity"] = quantity

        if quantity <= 1e-6:
            final_trade_decision["signal_action"] = "HOLD"; final_trade_decision["trade_action"] = "NO_ACTION"
            final_trade_decision["risk_reason"] = "Position size after risk assessment is zero."; return final_trade_decision

        dynamic_rc_boundaries = raw_signal.get("details", {}).get("dynamic_rc_boundaries")
        sl_price, tp_price = self._determine_stop_loss_take_profit(
            raw_signal["signal_action"], fhmv_state_name, entry_price, current_vol_atr,
            dynamic_rc_boundaries, raw_signal.get("details"))
        final_trade_decision["stop_loss_price"] = sl_price
        final_trade_decision["take_profit_price"] = tp_price
        
        final_trade_decision["risk_reason"] = "Trade signal passed risk assessment."
        if raw_signal["signal_action"] == "BUY": final_trade_decision["trade_action"] = "EXECUTE_BUY"
        elif raw_signal["signal_action"] == "SELL": final_trade_decision["trade_action"] = "EXECUTE_SELL"
        
        return final_trade_decision