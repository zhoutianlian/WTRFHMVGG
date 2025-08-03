# fhmv/visualization/visualization_module.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection # For state regions
import seaborn as sns

# Attempt to import constants, if not found, define locally for standalone use
try:
    from ..utils import constants
except ImportError:
    # Define constants locally if the package structure isn't fully set up during isolated execution
    class MockConstants:
        FHMV_REGIME_ST = "ST"; FHMV_REGIME_VT = "VT"; FHMV_REGIME_RC = "RC"
        FHMV_REGIME_RHA = "RHA"; FHMV_REGIME_HPEM = "HPEM"; FHMV_REGIME_AMB = "AMB"
        FHMV_REGIME_IDX_TO_NAME_MAP = {
            0: FHMV_REGIME_ST, 1: FHMV_REGIME_VT, 2: FHMV_REGIME_RC,
            3: FHMV_REGIME_RHA, 4: FHMV_REGIME_HPEM, 5: FHMV_REGIME_AMB
        }
    constants = MockConstants()


class VisualizationModule:
    """
    Provides methods for visualizing various aspects of the FHMV model pipeline.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the VisualizationModule.

        Args:
            config (dict, optional): Configuration dictionary for plot styles.
                                     Example: {"plot_style": "seaborn-v0_8-darkgrid",
                                                "figure_figsize": [15, 7],
                                                "regime_colors": {"ST": "blue", ...}}
        """
        self.config = config if config else {}
        self._setup_plot_style()

        # Define default colors for the 6 FHMV regimes for consistency
        self.regime_colors = self.config.get('regime_colors', {
            constants.FHMV_REGIME_ST: 'green',
            constants.FHMV_REGIME_VT: 'orange',
            constants.FHMV_REGIME_RC: 'cornflowerblue', # Light blue for range
            constants.FHMV_REGIME_RHA: 'purple',
            constants.FHMV_REGIME_HPEM: 'red',
            constants.FHMV_REGIME_AMB: 'grey'
        })
        # print("VisualizationModule Initialized.") # Removed for brevity

    def _setup_plot_style(self):
        """Applies a consistent style to plots."""
        try:
            plt.style.use(self.config.get('plot_style', 'seaborn-v0_8-v0_8-darkgrid'))
        except OSError:
            print(f"Warning: Plot style '{self.config.get('plot_style')}' not found. Using default.")
            plt.style.use('seaborn-v0_8-darkgrid') # A common, nice default

        plt.rcParams['figure.figsize'] = self.config.get('figure_figsize', (15, 7))
        plt.rcParams['axes.titlesize'] = self.config.get('axes_titlesize', 16)
        plt.rcParams['axes.labelsize'] = self.config.get('axes_labelsize', 12)
        plt.rcParams['xtick.labelsize'] = self.config.get('xtick_labelsize', 10)
        plt.rcParams['ytick.labelsize'] = self.config.get('ytick_labelsize', 10)
        plt.rcParams['legend.fontsize'] = self.config.get('legend_fontsize', 10)
        plt.rcParams['lines.linewidth'] = self.config.get('lines_linewidth', 1.5)
        plt.rcParams['axes.grid'] = self.config.get('axes_grid', True)
        plt.rcParams['figure.autolayout'] = True # Often helps with layout

    # --- Feature Visualization Functions ---
    def plot_feature_distribution(self, data: pd.Series, feature_name: str = None, 
                                  bins: int = 50, title: str = None, 
                                  kde: bool = True, hist: bool = True):
        """Plots the distribution (histogram and/or KDE) of a single feature."""
        if not isinstance(data, pd.Series):
            raise TypeError("Input 'data' must be a Pandas Series.")
        
        plt.figure()
        name = feature_name if feature_name else data.name
        plot_title = title if title else f"Distribution of {name}"
        
        if hist:
            sns.histplot(data, bins=bins, kde=False, label="Histogram", stat="density")
        if kde:
            sns.kdeplot(data, label="KDE", color='red' if hist else None, fill=True if hist else False, alpha=0.5 if hist else 1.0)
        
        plt.title(plot_title)
        plt.xlabel(name)
        plt.ylabel("Density" if hist else "Density")
        if hist and kde: plt.legend()
        plt.show()

    def plot_feature_time_series(self, data: pd.DataFrame, feature_names: list = None,
                                 title: str = "Feature Time Series", subplots: bool = False,
                                 date_format: str = "%Y-%m-%d %H:%M"):
        """Plots one or more features as time series."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a Pandas DataFrame with a DatetimeIndex.")
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                raise ValueError("Data index must be a DatetimeIndex or convertible to one.")

        if feature_names is None:
            feature_names = data.columns.tolist()
        
        num_features = len(feature_names)
        if num_features == 0:
            print("No features specified to plot.")
            return

        if subplots:
            fig, axes = plt.subplots(num_features, 1, sharex=True, 
                                     figsize=(self.config.get('figure_figsize', (15, 3 * num_features))))
            if num_features == 1: axes = [axes] # Ensure axes is always iterable
            fig.suptitle(title, fontsize=self.config.get('axes_titlesize', 16) + 2)
            for i, feature in enumerate(feature_names):
                if feature in data.columns:
                    axes[i].plot(data.index, data[feature], label=feature)
                    axes[i].set_ylabel(feature)
                    axes[i].legend(loc='upper left')
                else:
                    axes[i].text(0.5, 0.5, f"'{feature}' not found", ha='center', va='center')
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            plt.xlabel("Time")
        else:
            plt.figure()
            for feature in feature_names:
                if feature in data.columns:
                    plt.plot(data.index, data[feature], label=feature)
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend(loc='best')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        plt.show()

    def plot_ewma_smoothing_effect(self, raw_series: pd.Series, smoothed_series: pd.Series,
                                   series_name: str, title: str = None,
                                   date_format: str = "%Y-%m-%d %H:%M"):
        """Compares a raw time series with its EWMA-smoothed version."""
        if not (isinstance(raw_series, pd.Series) and isinstance(smoothed_series, pd.Series)):
            raise TypeError("Inputs must be Pandas Series.")
        if not isinstance(raw_series.index, pd.DatetimeIndex) or not isinstance(smoothed_series.index, pd.DatetimeIndex):
            raise ValueError("Series indices must be DatetimeIndex.")

        plt.figure()
        plot_title = title if title else f"EWMA Smoothing Effect on {series_name}"
        
        plt.plot(raw_series.index, raw_series, label=f"Raw {series_name}", alpha=0.7)
        plt.plot(smoothed_series.index, smoothed_series, label=f"EWMA Smoothed {series_name}", color='red')
        
        plt.title(plot_title)
        plt.xlabel("Time")
        plt.ylabel(series_name)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.show()

    # --- FHMV Core Engine Visualization Functions ---
    def plot_em_log_likelihood(self, log_likelihoods: list, title: str = "EM Log-Likelihood Convergence"):
        """Plots the log-likelihood over EM iterations."""
        if not isinstance(log_likelihoods, list) or not all(isinstance(x, (int, float)) for x in log_likelihoods):
            raise TypeError("log_likelihoods must be a list of numbers.")
            
        plt.figure()
        plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, marker='o', linestyle='-')
        plt.title(title)
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-Likelihood")
        plt.xticks(range(1, len(log_likelihoods) + 1))
        plt.show()

    def plot_decoded_fhmv_states(self, price_series: pd.Series, 
                                 state_sequence: np.ndarray, # Array of regime indices (0-5)
                                 title: str = "Decoded FHMV Regimes with Price",
                                 date_format: str = "%Y-%m-%d"):
        """
        Plots price series with colored background regions indicating the decoded FHMV state.
        """
        if not isinstance(price_series, pd.Series) or not isinstance(price_series.index, pd.DatetimeIndex):
            raise TypeError("price_series must be a Pandas Series with a DatetimeIndex.")
        if not isinstance(state_sequence, np.ndarray) or state_sequence.ndim != 1:
            raise TypeError("state_sequence must be a 1D NumPy array of regime indices.")
        if len(price_series) != len(state_sequence):
            # Attempt to align if state_sequence is for a subset (e.g. after dropping NaNs)
            if len(price_series.loc[price_series.index.isin(pd.Series(state_sequence, index=price_series.index[-len(state_sequence):]).index)]) == len(state_sequence):
                 price_series = price_series.loc[price_series.index.isin(pd.Series(state_sequence, index=price_series.index[-len(state_sequence):]).index)]
            else:
                 raise ValueError("Length of price_series and state_sequence must match, or be alignable.")


        fig, ax1 = plt.subplots(figsize=self.config.get('figure_figsize', (15,7)))
        
        # Plot price
        ax1.plot(price_series.index, price_series, color='black', label='Price', lw=1.5, alpha=0.9)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

        # Create a second y-axis for state indicators if needed, or just use background colors
        # For background colors, iterate through state changes
        unique_states = np.unique(state_sequence)
        # print(f"Unique states found in sequence: {unique_states}")
        
        # Create polygons for each state segment
        # Convert datetime index to numerical representation for fill_betweenx
        # x_numeric = mdates.date2num(price_series.index.to_pydatetime()) # Requires price_series.index
        
        current_state = state_sequence[0]
        start_idx = 0
        for i in range(1, len(state_sequence)):
            if state_sequence[i] != current_state or i == len(state_sequence) - 1:
                end_idx = i if state_sequence[i] != current_state else i + 1
                regime_name = constants.FHMV_REGIME_IDX_TO_NAME_MAP.get(current_state, f"State {current_state}")
                color = self.regime_colors.get(regime_name, 'gray') # Default color
                
                ax1.axvspan(price_series.index[start_idx], price_series.index[end_idx-1], 
                            facecolor=color, alpha=0.2, label=f"{regime_name}" if i == 1 else "_nolegend_") # Label only first segment for legend
                
                current_state = state_sequence[i]
                start_idx = i
        
        # Improve legend for regime colors
        handles, labels = ax1.get_legend_handles_labels()
        # Add custom patches for regime colors if axvspan labels are tricky
        patch_handles = []
        for state_idx, name in constants.FHMV_REGIME_IDX_TO_NAME_MAP.items():
            if state_idx in unique_states: # Only show legend for states present
                 patch_handles.append(plt.Rectangle((0,0),1,1, facecolor=self.regime_colors.get(name,'gray'), alpha=0.3, label=name))
        
        if patch_handles:
            ax1.legend(handles=handles + patch_handles, loc='best')

        plt.title(title)
        plt.show()

    def plot_fhmv_state_probabilities(self, state_probabilities_df: pd.DataFrame, # Index=time, Cols=regime names or indices
                                      title: str = "FHMV Regime Probabilities Over Time",
                                      date_format: str = "%Y-%m-%d"):
        """
        Plots the probabilities of being in each FHMV regime over time as a stacked area plot.
        state_probabilities_df: DataFrame with DatetimeIndex and columns for each regime's probability.
                                Column names should match keys in self.regime_colors or be mappable.
        """
        if not isinstance(state_probabilities_df, pd.DataFrame) or not isinstance(state_probabilities_df.index, pd.DatetimeIndex):
            raise TypeError("state_probabilities_df must be a Pandas DataFrame with a DatetimeIndex.")
        
        # Ensure columns match the order and names expected by regime_colors
        # Or dynamically create colors list based on columns present
        plot_columns = []
        plot_colors = []
        
        # Try to map columns to known regime names and their colors
        # First, check if columns are already regime names
        if all(col_name in self.regime_colors for col_name in state_probabilities_df.columns):
            plot_columns = [col for col in constants.FHMV_REGIME_NAMES if col in state_probabilities_df.columns] # Preserve order
            plot_colors = [self.regime_colors[col] for col in plot_columns]
        # Else, check if columns are integer indices
        elif all(isinstance(col, int) for col in state_probabilities_df.columns):
            temp_map = {constants.FHMV_REGIME_IDX_TO_NAME_MAP.get(col_idx): col_idx for col_idx in state_probabilities_df.columns}
            plot_columns_mapped = []
            for name_const in constants.FHMV_REGIME_NAMES: # Ensure consistent order
                if name_const in temp_map:
                    plot_columns_mapped.append(temp_map[name_const]) # Get the original column index
                    plot_colors.append(self.regime_colors[name_const])
            plot_columns = plot_columns_mapped
        else:
            print("Warning: Columns in state_probabilities_df do not directly match regime names or indices. Using default colors.")
            plot_columns = state_probabilities_df.columns.tolist()
            # Generate default colors if not matching
            palette = sns.color_palette("husl", len(plot_columns))
            plot_colors = [palette[i % len(palette)] for i in range(len(plot_columns))]

        if not plot_columns:
            print("No valid columns found in state_probabilities_df for plotting.")
            return

        plt.figure()
        # Use a subset of df for plotting if too many columns were generated but not found in mapping
        df_to_plot = state_probabilities_df[plot_columns]
        
        plt.stackplot(df_to_plot.index, 
                      *[df_to_plot[col] for col in df_to_plot.columns], # Ensure order of data matches labels/colors
                      labels=plot_columns, # These should be regime names if mapped
                      colors=plot_colors,
                      alpha=0.7)
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Probability")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(3, len(plot_columns)))
        plt.ylim(0, 1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for legend
        plt.show()

    # --- Signal Generation Visualization Functions (Placeholders) ---
    def plot_signals_on_price(self, price_series: pd.Series, 
                              buy_signals_idx: pd.DatetimeIndex = None, 
                              sell_signals_idx: pd.DatetimeIndex = None, 
                              title: str = "Trading Signals on Price",
                              date_format: str = "%Y-%m-%d"):
        """Plots price with buy/sell markers."""
        plt.figure()
        plt.plot(price_series.index, price_series, label="Price", color='black', alpha=0.8)
        
        if buy_signals_idx is not None and not buy_signals_idx.empty:
            # Ensure buy_signals_idx are present in price_series.index
            valid_buy_signals_idx = buy_signals_idx[buy_signals_idx.isin(price_series.index)]
            if not valid_buy_signals_idx.empty:
                 plt.scatter(valid_buy_signals_idx, price_series.loc[valid_buy_signals_idx], 
                            label='Buy Signal', marker='^', color='green', s=100, edgecolor='black', zorder=5)
        
        if sell_signals_idx is not None and not sell_signals_idx.empty:
            valid_sell_signals_idx = sell_signals_idx[sell_signals_idx.isin(price_series.index)]
            if not valid_sell_signals_idx.empty:
                plt.scatter(valid_sell_signals_idx, price_series.loc[valid_sell_signals_idx], 
                            label='Sell Signal', marker='v', color='red', s=100, edgecolor='black', zorder=5)
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.show()

    # --- Backtesting Visualization Functions (Placeholders) ---
    def plot_equity_curve(self, equity_curve_df: pd.DataFrame, title: str = "Portfolio Equity Curve", date_format: str = "%Y-%m-%d"):
        """Plots the portfolio equity over time."""
        if not isinstance(equity_curve_df, pd.DataFrame) or 'equity' not in equity_curve_df.columns:
            raise ValueError("equity_curve_df must be a DataFrame with an 'equity' column and DatetimeIndex.")
        if not isinstance(equity_curve_df.index, pd.DatetimeIndex):
            equity_curve_df.index = pd.to_datetime(equity_curve_df.index)

        plt.figure()
        plt.plot(equity_curve_df.index, equity_curve_df['equity'], label="Portfolio Equity")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.show()

    def plot_drawdown_series(self, equity_curve_df: pd.DataFrame, title: str = "Portfolio Drawdown Series (%)", date_format: str = "%Y-%m-%d"):
        """Calculates and plots the portfolio drawdown series."""
        if not isinstance(equity_curve_df, pd.DataFrame) or 'equity' not in equity_curve_df.columns:
            raise ValueError("equity_curve_df must be a DataFrame with an 'equity' column and DatetimeIndex.")
        if not isinstance(equity_curve_df.index, pd.DatetimeIndex):
            equity_curve_df.index = pd.to_datetime(equity_curve_df.index)
            
        peak = equity_curve_df['equity'].expanding(min_periods=1).max()
        drawdown_percent = ((equity_curve_df['equity'] - peak) / peak) * 100
        
        plt.figure()
        plt.fill_between(drawdown_percent.index, drawdown_percent, 0, 
                         color='red', alpha=0.3, label="Drawdown")
        # plt.plot(drawdown_percent.index, drawdown_percent, label="Drawdown (%)", color='red') # Line plot if preferred
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.show()