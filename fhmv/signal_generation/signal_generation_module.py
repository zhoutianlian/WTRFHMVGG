# fhmv/signals/signal_generation_module.py

import pandas as pd
import numpy as np
from .clad_signal_generator import CLADSignalGenerator # Relative import
from .l3_signal_assessor import L3SignalAssessor     # Relative import

# Assuming FHMV_REGIME constants are defined, e.g., in fhmv.utils.constants
# For standalone, define them here or ensure they are passed/accessible.
FHMV_REGIME_ST = "ST"; FHMV_REGIME_VT = "VT"; FHMV_REGIME_RC = "RC"
FHMV_REGIME_RHA = "RHA"; FHMV_REGIME_HPEM = "HPEM"; FHMV_REGIME_AMB = "AMB"


class SignalGenerationModule:
    """
    FHMV Signal Generation Module.
    Orchestrates CLAD and L3 signal generation based on FHMV state.
   
    """
    def __init__(self, config: dict):
        """
        Initializes the SignalGenerationModule with configurations for its sub-components.
        """
        self.clad_parameters = config.get("clad_parameters", {})
        l3_params = config.get("l3_parameters", {})
        non_rc_l3_rules = config.get("non_rc_l3_rules", {})
        self.rc_boundary_params = config.get("rc_boundary_params", {})
        
        self.clad_signal_generator = CLADSignalGenerator(self.clad_parameters)
        self.l3_signal_assessor = L3SignalAssessor(l3_params, non_rc_l3_rules)
        
        self.regime_names_map_idx_to_str = { # Example mapping
            0: FHMV_REGIME_ST, 1: FHMV_REGIME_VT, 2: FHMV_REGIME_RC,
            3: FHMV_REGIME_RHA, 4: FHMV_REGIME_HPEM, 5: FHMV_REGIME_AMB
        }
        print("SignalGenerationModule Initialized with config.")

    def _calculate_dynamic_rc_boundaries(self, price_history: pd.Series) -> dict:
        """ Dynamic RC boundary calculation logic. """
        # (Placeholder logic from previous step)
        if price_history.empty or len(price_history) < self.rc_boundary_params.get('min_window', 5): # Reduced min_window for less data
            return {"upper": np.nan, "lower": np.nan}
        
        window = self.rc_boundary_params.get('window', 20) # Reduced window
        std_multiplier = self.rc_boundary_params.get('std_multiplier', 2.0)
        
        # Ensure enough points for rolling, use min_periods
        min_roll_periods = max(1, window // 2, self.rc_boundary_params.get('min_window', 5))

        rolling_mean = price_history.rolling(window=window, min_periods=min_roll_periods).mean()
        rolling_std = price_history.rolling(window=window, min_periods=min_roll_periods).std()
        
        if rolling_mean.empty or pd.isna(rolling_mean.iloc[-1]) or \
           rolling_std.empty or pd.isna(rolling_std.iloc[-1]):
            return {"upper": np.nan, "lower": np.nan}

        upper_boundary = rolling_mean.iloc[-1] + std_multiplier * rolling_std.iloc[-1]
        lower_boundary = rolling_mean.iloc[-1] - std_multiplier * rolling_std.iloc[-1]
        
        return {"upper": upper_boundary, "lower": lower_boundary}

    def generate_signal(self, current_fhmv_state_idx: int,
                          current_processed_features: pd.Series,
                          recent_features_history: pd.DataFrame) -> dict:
        # (Main dispatch logic from previous step, ensures proper method calls)
        timestamp = current_processed_features.name if hasattr(current_processed_features, 'name') else pd.Timestamp.now()
        asset_id = "DEFAULT_ASSET" 
        current_fhmv_state_name = self.regime_names_map_idx_to_str.get(current_fhmv_state_idx, FHMV_REGIME_AMB)

        signal_action = "HOLD"; strategy_name = f"{current_fhmv_state_name}_NO_ACTION"; confidence = 0.0
        details = {"fhmv_state": current_fhmv_state_name, "reason": "Default no action."}

        if current_fhmv_state_name == FHMV_REGIME_AMB:
            strategy_name = "AMB_PROTOCOL"; confidence = 1.0; details["reason"] = "AMB state active."
        elif current_fhmv_state_name == FHMV_REGIME_RC:
            dynamic_rc_boundaries = self._calculate_dynamic_rc_boundaries(recent_features_history['P'])
            details["dynamic_rc_boundaries"] = dynamic_rc_boundaries # Pass to risk module if needed
            if np.isnan(dynamic_rc_boundaries.get("upper")):
                 details["reason"] = "Insufficient data for RC boundaries."
                 strategy_name = "RC_NO_BOUNDARIES"
            else:
                clad_result = self.clad_signal_generator.generate_clad_signal(
                    current_processed_features, recent_features_history, dynamic_rc_boundaries)
                if clad_result["action"] != "NONE":
                    signal_action = clad_result["action"]; confidence = clad_result["confidence_score"]
                    strategy_name = f"CLAD_RC_{clad_result['details'].get('tested_boundary', 'N/A')}"
                    details.update(clad_result["details"])
                else: details["reason"] = "RC state, no CLAD signal."; strategy_name = "RC_NO_CLAD"
        elif current_fhmv_state_name in [FHMV_REGIME_RHA, FHMV_REGIME_VT]:
            delta_series = recent_features_history['Delta']
            l3_result = self.l3_signal_assessor.assess_l3_signal(
                current_fhmv_state_name, delta_series, current_processed_features, recent_features_history)
            if l3_result["action"] != "NONE":
                signal_action = l3_result["action"]; confidence = l3_result["confidence_score"]
                strategy_name = f"L3_{current_fhmv_state_name}_{l3_result['details'].get('confidence_category', 'LOW')}"
                details.update(l3_result["details"])
            else: details["reason"] = f"{current_fhmv_state_name}, no L3 signal."; strategy_name = f"{current_fhmv_state_name}_NO_L3"
        elif current_fhmv_state_name == FHMV_REGIME_ST:
            details["reason"] = "ST state, no specific strategy defined."; strategy_name = "ST_HOLD"
        elif current_fhmv_state_name == FHMV_REGIME_HPEM:
            details["reason"] = "HPEM state, no entry strategy defined."; strategy_name = "HPEM_HOLD"
        
        return {"timestamp": timestamp, "asset_id": asset_id, "signal_action": signal_action,
                "strategy_name": strategy_name, "confidence": confidence, "details": details}