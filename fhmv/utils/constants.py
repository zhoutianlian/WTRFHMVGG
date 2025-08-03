# fhmv/utils/constants.py

"""
Shared constants for the FHMV project.
"""

# FHMV Regime Names
FHMV_REGIME_ST = "ST"  # Stable Trend
FHMV_REGIME_VT = "VT"  # Volatile Trend
FHMV_REGIME_RC = "RC"  # Range-bound/Consolidation
FHMV_REGIME_RHA = "RHA" # Reversal/High Absorption
FHMV_REGIME_HPEM = "HPEM" # High Pressure/Extreme Move
FHMV_REGIME_AMB = "AMB" # Ambiguous

FHMV_REGIME_NAMES = [
    FHMV_REGIME_ST, FHMV_REGIME_VT, FHMV_REGIME_RC,
    FHMV_REGIME_RHA, FHMV_REGIME_HPEM, FHMV_REGIME_AMB
]

# Mapping from regime index (0-5) to name, assuming a fixed order.
# This can be used by modules if they receive integer state indices.
FHMV_REGIME_IDX_TO_NAME_MAP = {
    0: FHMV_REGIME_ST,
    1: FHMV_REGIME_VT,
    2: FHMV_REGIME_RC,
    3: FHMV_REGIME_RHA,
    4: FHMV_REGIME_HPEM,
    5: FHMV_REGIME_AMB
}
FHMV_REGIME_NAME_TO_IDX_MAP = {v: k for k, v in FHMV_REGIME_IDX_TO_NAME_MAP.items()}


# Signal Actions (from SignalGenerationModule)
SIGNAL_ACTION_BUY = "BUY"
SIGNAL_ACTION_SELL = "SELL"
SIGNAL_ACTION_HOLD = "HOLD"
SIGNAL_ACTION_NONE = "NONE" # Or "NO_SIGNAL"

# Trade Actions (from RiskManagementModule for ExecutionModule)
TRADE_ACTION_EXECUTE_BUY = "EXECUTE_BUY"
TRADE_ACTION_EXECUTE_SELL = "EXECUTE_SELL" # Could be to open short or add to short
TRADE_ACTION_CLOSE_POSITION = "CLOSE_POSITION" # Explicit close signal
TRADE_ACTION_NO_ACTION = "NO_ACTION"

# CLAD Boundary Constants (used in CLADSignalGenerator & SignalGenerationModule)
CLAD_UPPER_BOUNDARY = "UPPER"
CLAD_LOWER_BOUNDARY = "LOWER"

# HPEM Sub-Type Directions/Possibilities
HPEM_DIRECTION_BULLISH = "BULLISH"
HPEM_DIRECTION_BEARISH = "BEARISH"
HPEM_DIRECTION_UNDETERMINED = "UNDETERMINED"

HPEM_SQUEEZE_POSSIBILITY_HIGH = "HIGH"
HPEM_SQUEEZE_POSSIBILITY_MEDIUM = "MEDIUM"
HPEM_SQUEEZE_POSSIBILITY_LOW = "LOW"
HPEM_SQUEEZE_NOT_APPLICABLE = "NOT_APPLICABLE"


# Default Asset ID (used in BacktestingModule and potentially SignalGenerationModule)
DEFAULT_ASSET_ID = "DEFAULT_ASSET"

# Small epsilon for numerical stability in various calculations
EPSILON_FLOAT = 1e-9
EPSILON_LOG = 1e-30 # For np.log(x + EPSILON_LOG)