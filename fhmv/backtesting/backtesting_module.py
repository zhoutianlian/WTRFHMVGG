# fhmv/backtesting/backtesting_module.py

import pandas as pd
import numpy as np
from datetime import datetime

# It's assumed that the actual module classes would be imported if this was run
# For example:
# from fhmv.data_processing.data_ingestion_preprocessing_module import DataIngestionPreprocessingModule
# from fhmv.feature_engineering.feature_engineering_module import FeatureEngineeringModule
# from fhmv.core_engine.fhmv_core_engine import FHMVCoreEngine
# from fhmv.signal_generation.signal_generation_module import SignalGenerationModule
# from fhmv.risk_management.risk_management_module import RiskManagementModule

# Define FHMV Regime constants if not imported from a central place (e.g. fhmv.utils.constants)
FHMV_REGIME_ST = "ST"; FHMV_REGIME_VT = "VT"; FHMV_REGIME_RC = "RC"
FHMV_REGIME_RHA = "RHA"; FHMV_REGIME_HPEM = "HPEM"; FHMV_REGIME_AMB = "AMB"
DEFAULT_ASSET_ID = "DEFAULT_ASSET" # Example constant

class BacktestingModule:
    """
    FHMV Execution Simulation and Backtesting Module.
    (Full docstring from previous step)
    """

    def __init__(self,
                 config: dict, # Expects a key "backtesting_config"
                 data_ingestion_module, 
                 feature_engineering_module, 
                 fhmv_core_engine,          
                 signal_generation_module,   
                 risk_management_module):    
        """
        Initializes the BacktestingModule.
        (Full docstring from previous step)
        """
        backtest_cfg = config.get("backtesting_config", {})
        self.initial_capital = float(backtest_cfg.get("initial_capital", 100000.0))
        self.commission_rate = float(backtest_cfg.get("commission_rate_per_trade", 0.001))
        self.slippage_factor = float(backtest_cfg.get("slippage_per_trade_factor", 0.0005))

        self.data_ingestion_module = data_ingestion_module
        self.feature_engineering_module = feature_engineering_module
        self.fhmv_core_engine = fhmv_core_engine
        self.signal_generation_module = signal_generation_module
        self.risk_management_module = risk_management_module

        self.equity_curve = [] 
        self.trade_log = [] 
        self.reset_backtest_state() # Initialize cash etc.
        # print("BacktestingModule Initialized.") # Removed for brevity

    def reset_backtest_state(self):
        """Resets the state of the backtest for a new run."""
        self.cash = self.initial_capital
        self.current_position_units = 0.0 
        self.current_position_avg_price = 0.0 
        self.equity_curve = [{"timestamp": None, "equity": self.initial_capital}] # Start with initial capital
        self.trade_log = [] 
        # self.fhmv_regime_at_trade = [] # No longer used directly here, logged in trade_log

    def _process_trade_decision(self, timestamp: datetime, current_price: float,
                                  trade_decision: dict, current_fhmv_regime_idx: int):
        """Processes a trade decision, updates portfolio, and logs the trade."""
        # (Code from previous step, ensuring robustness and clarity)
        action = trade_decision.get("trade_action") # EXECUTE_BUY, EXECUTE_SELL, CLOSE_POSITION, NO_ACTION
        quantity = trade_decision.get("risk_adjusted_quantity", 0.0)
        
        trade_executed_this_step = False
        pnl_realized_this_step = 0.0
        commission_paid_this_step = 0.0
        asset_id = trade_decision.get("asset_id", DEFAULT_ASSET_ID)

        # --- Handling Closing Existing Position Before New Trade / Explicit Close ---
        should_close_existing = False
        if action == "CLOSE_POSITION" and self.current_position_units != 0:
            should_close_existing = True
        elif (action == "EXECUTE_SELL" and self.current_position_units > 0) or \
             (action == "EXECUTE_BUY" and self.current_position_units < 0): # Implied close to reverse
            should_close_existing = True

        if should_close_existing:
            close_qty = abs(self.current_position_units)
            direction_of_close = "SELL_TO_CLOSE" if self.current_position_units > 0 else "BUY_TO_COVER"
            
            slippage_adj = -self.slippage_factor if direction_of_close == "SELL_TO_CLOSE" else self.slippage_factor
            close_price_with_slippage = current_price * (1 + slippage_adj)
            
            value_at_close = close_qty * close_price_with_slippage
            commission_on_close = value_at_close * self.commission_rate
            self.cash -= commission_on_close
            
            if self.current_position_units > 0: # Closing a long
                self.cash += value_at_close
                pnl_realized_this_step = close_qty * (close_price_with_slippage - self.current_position_avg_price)
            else: # Closing a short
                self.cash -= value_at_close # Buy back value subtracted from cash (initial short proceeds already added)
                pnl_realized_this_step = close_qty * (self.current_position_avg_price - close_price_with_slippage)
            
            pnl_realized_this_step -= commission_on_close # Net PnL includes commission

            self.trade_log.append({
                "timestamp": timestamp, "asset_id": asset_id, "type": "CLOSE",
                "direction": direction_of_close, "quantity": close_qty, 
                "price": close_price_with_slippage, "commission": commission_on_close, 
                "pnl": pnl_realized_this_step, "fhmv_regime_idx": current_fhmv_regime_idx,
                "signal_details": trade_decision.get("details", {})
            })
            self.current_position_units = 0.0
            self.current_position_avg_price = 0.0
            trade_executed_this_step = True

        # --- Handling New Opening Trade Signal (if not just a close) ---
        if action == "EXECUTE_BUY" and quantity > 0:
            exec_price = current_price * (1 + self.slippage_factor)
            cost_of_trade = quantity * exec_price
            commission_paid = cost_of_trade * self.commission_rate
            total_outlay = cost_of_trade + commission_paid

            if self.cash >= total_outlay:
                self.cash -= total_outlay
                new_total_units = self.current_position_units + quantity
                self.current_position_avg_price = ((self.current_position_avg_price * self.current_position_units) + (exec_price * quantity)) / new_total_units if self.current_position_units > 0 else exec_price
                self.current_position_units = new_total_units
                trade_executed_this_step = True
                self.trade_log.append({
                    "timestamp": timestamp, "asset_id": asset_id, "type": "OPEN_LONG",
                    "direction": "BUY", "quantity": quantity, "price": exec_price, 
                    "commission": commission_paid, "pnl": 0, # PnL realized on close
                    "fhmv_regime_idx": current_fhmv_regime_idx,
                    "signal_details": trade_decision.get("details", {})
                })
        elif action == "EXECUTE_SELL" and quantity > 0: # Opening a short
            exec_price = current_price * (1 - self.slippage_factor)
            proceeds_from_trade = quantity * exec_price
            commission_paid = proceeds_from_trade * self.commission_rate
            
            self.cash += (proceeds_from_trade - commission_paid)
            new_total_units = self.current_position_units - quantity # Units become more negative
            # Avg price for shorts: (total value shorted) / (total units shorted)
            if self.current_position_units < 0: # Adding to existing short
                 self.current_position_avg_price = ((self.current_position_avg_price * abs(self.current_position_units)) + (exec_price * quantity)) / (abs(self.current_position_units) + quantity)
            else: # New short
                self.current_position_avg_price = exec_price
            self.current_position_units = new_total_units
            trade_executed_this_step = True
            self.trade_log.append({
                "timestamp": timestamp, "asset_id": asset_id, "type": "OPEN_SHORT",
                "direction": "SELL_SHORT", "quantity": quantity, "price": exec_price, 
                "commission": commission_paid, "pnl": 0, 
                "fhmv_regime_idx": current_fhmv_regime_idx,
                "signal_details": trade_decision.get("details", {})
            })
        # Note: Stop-loss/take-profit orders from trade_decision are for information here;
        # actual SL/TP execution would be checked per bar against H/L prices.

    def _update_portfolio_value(self, timestamp: datetime, current_price: float):
        """Calculates current portfolio value and records it."""
        market_value_of_positions = self.current_position_units * current_price
        portfolio_value = self.cash + market_value_of_positions
        # Avoid appending if timestamp is the same as last to prevent duplicate equity points for initial state
        if not self.equity_curve or self.equity_curve[-1]["timestamp"] != timestamp:
            self.equity_curve.append({"timestamp": timestamp, "equity": portfolio_value})
        elif self.equity_curve[-1]["timestamp"] == timestamp : # Update if same timestamp
             self.equity_curve[-1]["equity"] = portfolio_value


    def _calculate_performance_metrics(self) -> dict:
        """Calculates comprehensive performance metrics."""
        # (Code from previous step, with minor robustness and structure)
        if len(self.equity_curve) < 2: # Need at least initial and one more point
            return {"error": "Not enough equity data points to calculate metrics."}

        equity_df = pd.DataFrame(self.equity_curve)
        # Drop first entry if it was placeholder for initial capital before first timestamp
        if equity_df.iloc[0]['timestamp'] is None : equity_df = equity_df.iloc[1:]
        if equity_df.empty : return {"error": "Equity curve DataFrame empty after init drop."}
        equity_df = equity_df.set_index("timestamp")


        final_equity = equity_df['equity'].iloc[-1]
        total_return_percent = (final_equity / self.initial_capital - 1) * 100
        
        peak = equity_df['equity'].expanding(min_periods=1).max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown_percent = drawdown.min() * 100 if not drawdown.empty else 0.0

        equity_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = np.nan
        if not equity_returns.empty and equity_returns.std() != 0:
            # Placeholder: This annualization needs to match data frequency (e.g., hourly)
            # For hourly: 252 trading days * X hours/day. Assuming X=8 for example.
            trading_periods_per_year = 252 * 8 
            if len(equity_returns) > 20: # Min data points for meaningful std dev
                annualized_mean_return = equity_returns.mean() * trading_periods_per_year
                annualized_std_return = equity_returns.std() * np.sqrt(trading_periods_per_year)
                if annualized_std_return > 1e-9:
                    sharpe_ratio = annualized_mean_return / annualized_std_return # Assuming Rf=0
        
        # Trade Stats
        entry_trades = [t for t in self.trade_log if t['type'] in ['OPEN_LONG', 'OPEN_SHORT']]
        closing_trades_with_pnl = [t for t in self.trade_log if t['type'] in ['CLOSE', 'CLOSE_EXPLICIT']] # 'CLOSE' is from reversal
        
        num_trades_opened = len(entry_trades)
        num_trades_closed = len(closing_trades_with_pnl)
        
        profitable_pnls = [t['pnl'] for t in closing_trades_with_pnl if t['pnl'] > 0]
        losing_pnls = [t['pnl'] for t in closing_trades_with_pnl if t['pnl'] < 0] # PnL already includes commission
        
        num_wins = len(profitable_pnls)
        num_losses = len(losing_pnls)
        win_rate_percent = (num_wins / num_trades_closed * 100) if num_trades_closed > 0 else 0.0
        
        avg_profit_per_win = np.mean(profitable_pnls) if num_wins > 0 else 0.0
        avg_loss_per_loss = np.mean(losing_pnls) if num_losses > 0 else 0.0 # Will be negative
        profit_factor = np.sum(profitable_pnls) / abs(np.sum(losing_pnls)) if abs(np.sum(losing_pnls)) > 0 else np.inf

        return {
            "initial_capital": self.initial_capital, "final_equity": final_equity,
            "total_return_percent": total_return_percent, "max_drawdown_percent": max_drawdown_percent,
            "sharpe_ratio_annualized_approx": sharpe_ratio,
            "total_opening_trades": num_trades_opened, "total_closing_trades": num_trades_closed,
            "num_winning_trades": num_wins, "num_losing_trades": num_losses,
            "win_rate_percent": win_rate_percent,
            "avg_profit_per_winning_trade": avg_profit_per_win,
            "avg_loss_per_losing_trade": avg_loss_per_loss,
            "profit_factor": profit_factor
        }

    def _generate_report(self, performance_metrics: dict, regime_names_map: dict = None) -> str:
        """ Generates a summary report of the backtest results. """
        # (Code from previous step, adding regime stratification placeholder)
        report_lines = ["--- FHMV Backtest Report ---"]
        for key, value in performance_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        report_lines.append("\n--- Regime-Stratified Statistics (Placeholder) ---")
        if regime_names_map and self.trade_log:
            trades_df = pd.DataFrame(self.trade_log)
            closing_trades_df = trades_df[trades_df['type'].isin(['CLOSE', 'CLOSE_EXPLICIT'])].copy()
            if not closing_trades_df.empty:
                for regime_idx, regime_name in regime_names_map.items():
                    regime_trades = closing_trades_df[closing_trades_df['fhmv_regime_idx'] == regime_idx]
                    if not regime_trades.empty:
                        regime_pnl = regime_trades['pnl'].sum()
                        regime_wins = len(regime_trades[regime_trades['pnl'] > 0])
                        regime_win_rate = (regime_wins / len(regime_trades) * 100) if len(regime_trades) > 0 else 0
                        report_lines.append(f"Regime {regime_name} (Idx {regime_idx}): #Trades Closed={len(regime_trades)}, Total PnL={regime_pnl:.2f}, WinRate={regime_win_rate:.2f}%")
                    else:
                        report_lines.append(f"Regime {regime_name} (Idx {regime_idx}): No closing trades.")
            else:
                report_lines.append("No closing trades to stratify by regime.")

        report_lines.append("\n--- Sample Trade Log (Max 10 Entries) ---")
        for i, entry in enumerate(self.trade_log):
            if i >= 10: break
            report_lines.append(f"  {entry}")
        return "\n".join(report_lines)


    def run_backtest(self, historical_raw_data: pd.DataFrame):
        """Runs the event-driven backtest over the historical data."""
        # (Code from previous step, ensuring modules are used correctly)
        if self.fhmv_core_engine.stmm_emission_params_ is None: 
            raise RuntimeError("FHMV Core Engine must be trained before backtesting.")
        if self.feature_engineering_module.feature_means_ is None:
            raise RuntimeError("FeatureEngineeringModule must be fitted before backtesting.")

        self.reset_backtest_state()
        
        print("Backtest: Pre-calculating all features...")
        core_features_df = self.data_ingestion_module.process_raw_data(historical_raw_data)
        processed_features_df = self.feature_engineering_module.transform(core_features_df)
        
        first_valid_index = processed_features_df.first_valid_index()
        if first_valid_index is None:
            print("Error: No valid data after feature processing. Backtest cannot run."); return None
            
        # Use .loc for robust slicing even with non-sequential or gappy DatetimeIndex
        historical_raw_data_filtered = historical_raw_data.loc[first_valid_index:]
        processed_features_df_filtered = processed_features_df.loc[first_valid_index:]
        
        if processed_features_df_filtered.empty:
            print("Error: No data remains after filtering initial NaNs. Backtest cannot run."); return None

        print("Backtest: Performing FHMV state inference for the entire period...")
        # Get Viterbi path of final regime indices once
        all_viterbi_path_final_regimes = self.fhmv_core_engine.get_most_likely_state_sequence(
            processed_features_df_filtered # Pass the whole filtered series
        )
        if len(all_viterbi_path_final_regimes) != len(processed_features_df_filtered):
            raise ValueError("Length of Viterbi path does not match length of features for backtest.")

        print(f"Backtest: Starting event loop from {processed_features_df_filtered.index[0]} for {len(processed_features_df_filtered)} periods...")

        for t_idx in range(len(processed_features_df_filtered)):
            current_timestamp = processed_features_df_filtered.index[t_idx]
            current_market_price = historical_raw_data_filtered.loc[current_timestamp, 'close']
            current_fhmv_regime_idx = all_viterbi_path_final_regimes[t_idx]
            
            # History for signal gen / risk mgmt (rolling window ending at current t_idx)
            # Ensure this history is from processed_features_df_filtered to match inference
            history_window_size_sg = self.signal_generation_module.rc_boundary_params.get('window',50) + 5 
            start_hist_idx_sg = max(0, t_idx + 1 - history_window_size_sg)
            recent_history_for_signal_gen = processed_features_df_filtered.iloc[start_hist_idx_sg : t_idx+1]
            current_features_series_for_signal_gen = processed_features_df_filtered.iloc[t_idx]

            raw_signal = self.signal_generation_module.generate_signal(
                current_fhmv_regime_idx, current_features_series_for_signal_gen, recent_history_for_signal_gen)

            current_portfolio_status_dummy = {"cash": self.cash, "current_units": self.current_position_units, 
                                              "avg_entry_price": self.current_position_avg_price} 
            final_trade_decision = self.risk_management_module.manage_trade_signal(
                raw_signal, current_fhmv_regime_idx, current_features_series_for_signal_gen, 
                recent_history_for_signal_gen, # History for HPEM P_prev, RC boundaries
                current_portfolio_status_dummy)
            
            self._process_trade_decision(current_timestamp, current_market_price, 
                                         final_trade_decision, current_fhmv_regime_idx)
            
            self._update_portfolio_value(current_timestamp, current_market_price)
            if t_idx % 1000 == 0 and t_idx >0 : print(f"  Backtest processed {t_idx} periods...")


        performance_metrics = self._calculate_performance_metrics()
        report = self._generate_report(performance_metrics, 
                                       self.signal_generation_module.regime_names_map_idx_to_str) # Pass regime map
        print(report)

        return performance_metrics, self.trade_log, self.equity_curve