"""
RPN Features Visualizations

Professional visualizations for Risk Priority Number features including
CWT detrending, Kalman filtering, and correlation analysis.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .config import COLORS, CHART_STYLE, LAYOUT_TEMPLATES, get_color_with_alpha, get_trend_color


class RPNVisualizer:
    """Professional visualizations for RPN features"""
    
    def __init__(self, df: pd.DataFrame, time_col: str = 'time'):
        self.df = df.copy()
        self.time_col = time_col
        
        # Ensure time column is datetime
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            self.df = self.df.sort_values(time_col).reset_index(drop=True)
        
        self.layout_theme = LAYOUT_TEMPLATES['dark_financial']
    
    def create_rpn_dominance_plot(self) -> go.Figure:
        """Create comprehensive RPN and dominance analysis plot"""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=[
                'BTC Price with Market Dominance',
                'Risk Priority Number & Components', 
                'Dominance Classification & Signals'
            ]
        )
        
        # Row 1: Price with dominance coloring
        if 'price' in self.df.columns:
            # Base price line
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['price'],
                    mode='lines',
                    name='BTC Price',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add dominance overlays
            if 'dominance' in self.df.columns:
                # Bull dominance periods
                bull_mask = self.df['dominance'] == 1
                if bull_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=self.df[self.time_col][bull_mask],
                            y=self.df['price'][bull_mask],
                            mode='markers',
                            name='Bull Dominance',
                            marker={
                                'color': COLORS['success'],
                                'size': CHART_STYLE['marker_size_small'],
                                'symbol': 'triangle-up'
                            },
                            hovertemplate='Bull Dominance<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                # Bear dominance periods
                bear_mask = self.df['dominance'] == -1
                if bear_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=self.df[self.time_col][bear_mask],
                            y=self.df['price'][bear_mask],
                            mode='markers',
                            name='Bear Dominance',
                            marker={
                                'color': COLORS['danger'],
                                'size': CHART_STYLE['marker_size_small'],
                                'symbol': 'triangle-down'
                            },
                            hovertemplate='Bear Dominance<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # Row 2: RPN components
        rpn_components = ['risk_priority_number', 'lld_cwt_kf', 'fll_cwt_kf', 'fsl_cwt_kf']
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['warning'], COLORS['info']]
        
        for i, (component, color) in enumerate(zip(rpn_components, colors)):
            if component in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[component],
                        mode='lines',
                        name=component.replace('_', ' ').title(),
                        line={'color': color, 'width': CHART_STYLE['line_width_thin']},
                        hovertemplate=f'{component}: %{{y:.4f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Row 3: Dominance analysis
        if 'dominance' in self.df.columns:
            # Dominance line
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['dominance'],
                    mode='lines+markers',
                    name='Dominance',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width']},
                    marker={'size': CHART_STYLE['marker_size_small']},
                    hovertemplate='Dominance: %{y}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Add reference lines for dominance levels
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS['success'], 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=-1, line_dash="dash", line_color=COLORS['danger'], 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], 
                         row=3, col=1)
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title={
                'text': 'Risk Priority Number & Market Dominance Analysis',
                'font': {'size': CHART_STYLE['title_size']},
                'x': 0.5
            },
            height=CHART_STYLE['dashboard_height'],
            showlegend=True,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02}
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RPN Values", row=2, col=1)
        fig.update_yaxes(title_text="Dominance", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        return fig
    
    def create_cwt_analysis_plot(self) -> go.Figure:
        """Create CWT detrending analysis visualization"""
        
        # Find CWT-related columns
        original_cols = [col for col in self.df.columns if '_normal' in col]
        cwt_cols = [col for col in self.df.columns if '_cwt' in col and '_kf' not in col]
        
        if not original_cols or not cwt_cols:
            return self._create_empty_figure("No CWT data available")
        
        n_pairs = min(len(original_cols), len(cwt_cols))
        
        fig = make_subplots(
            rows=n_pairs, cols=2,
            shared_xaxes=True,
            subplot_titles=[f'Original vs CWT: {col.replace("_normal", "")}' for col in original_cols[:n_pairs]] * 2,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, (orig_col, cwt_col) in enumerate(zip(original_cols[:n_pairs], cwt_cols[:n_pairs])):
            row = i + 1
            
            # Original data
            if orig_col in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[orig_col],
                        mode='lines',
                        name=f'Original {orig_col}',
                        line={'color': COLORS['text_muted'], 'width': CHART_STYLE['line_width_thin']},
                        hovertemplate=f'{orig_col}: %{{y:.4f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=row, col=1
                )
            
            # CWT detrended data
            if cwt_col in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[cwt_col],
                        mode='lines',
                        name=f'CWT {cwt_col}',
                        line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                        hovertemplate=f'{cwt_col}: %{{y:.4f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=row, col=2
                )
                
                # Add trend line
                ma = self.df[cwt_col].rolling(window=24).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=ma,
                        mode='lines',
                        name=f'Trend {cwt_col}',
                        line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width_thin'], 'dash': 'dash'},
                        hovertemplate=f'Trend: %{{y:.4f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=row, col=2
                )
        
        fig.update_layout(
            **self.layout_theme,
            title='Continuous Wavelet Transform Analysis',
            height=max(600, n_pairs * 200),
            showlegend=True
        )
        
        return fig
    
    def create_kalman_analysis_plot(self) -> go.Figure:
        """Create Kalman filtering analysis visualization"""
        
        # Find Kalman-related columns
        cwt_cols = [col for col in self.df.columns if '_cwt' in col and '_kf' not in col]
        kf_cols = [col for col in self.df.columns if '_cwt_kf' in col]
        
        if not cwt_cols or not kf_cols:
            return self._create_empty_figure("No Kalman filter data available")
        
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'CWT vs Kalman Filtered Data',
                'Kalman Filter Innovation (Error)',
                'Liquidation Analysis (FLL vs FSL)',
                'Differential Analysis'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: CWT vs Kalman comparison
        for cwt_col, kf_col in zip(cwt_cols[:3], kf_cols[:3]):
            if cwt_col in self.df.columns and kf_col in self.df.columns:
                # CWT data
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[cwt_col],
                        mode='lines',
                        name=f'CWT {cwt_col.replace("_cwt", "")}',
                        line={'width': CHART_STYLE['line_width_thin']},
                        opacity=0.7,
                        hovertemplate=f'{cwt_col}: %{{y:.4f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Kalman filtered data
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[kf_col],
                        mode='lines',
                        name=f'KF {kf_col.replace("_cwt_kf", "")}',
                        line={'width': CHART_STYLE['line_width']},
                        hovertemplate=f'{kf_col}: %{{y:.4f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Innovation analysis (if available)
        if 'kalman_slope_innovation' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['kalman_slope_innovation'],
                    mode='lines',
                    name='Kalman Innovation',
                    line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Innovation: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: FLL vs FSL analysis
        if 'fll_cwt_kf' in self.df.columns and 'fsl_cwt_kf' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['fll_cwt_kf'],
                    mode='lines',
                    name='FLL (Kalman)',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='FLL: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['fsl_cwt_kf'],
                    mode='lines',
                    name='FSL (Kalman)',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='FSL: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Differential analysis
        if 'diff_ls_cwt_kf' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['diff_ls_cwt_kf'],
                    mode='lines',
                    name='FLL - FSL Differential',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['tertiary'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Differential: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add zero reference line
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], 
                         row=2, col=2)
        
        fig.update_layout(
            **self.layout_theme,
            title='Kalman Filter Analysis',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_correlation_analysis_plot(self) -> go.Figure:
        """Create correlation analysis visualization"""
        
        if 'corr_case' not in self.df.columns:
            return self._create_empty_figure("No correlation analysis data available")
        
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'FLL vs FSL Correlation Pattern',
                'Delta Analysis (Rate of Change)',
                'Correlation Case Distribution',
                'Liquidation Balance Over Time'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: FLL vs FSL scatter with correlation cases
        if 'fll_cwt_kf' in self.df.columns and 'fsl_cwt_kf' in self.df.columns:
            # Color by correlation case
            corr_cases = self.df['corr_case'].unique()
            case_colors = {
                'LUSUL': COLORS['success'], 'LUSUS': COLORS['primary'],
                'LDSDL': COLORS['danger'], 'LDSDS': COLORS['warning'],
                'LUSDL': COLORS['info'], 'LUSDS': COLORS['secondary'],
                'LDSUL': COLORS['tertiary'], 'LDSUS': '#FF9500',
                'None': COLORS['text_muted']
            }
            
            for case in corr_cases:
                if case != 'None':
                    mask = self.df['corr_case'] == case
                    if mask.any():
                        fig.add_trace(
                            go.Scatter(
                                x=self.df['fll_cwt_kf'][mask],
                                y=self.df['fsl_cwt_kf'][mask],
                                mode='markers',
                                name=case,
                                marker={
                                    'color': case_colors.get(case, COLORS['primary']),
                                    'size': CHART_STYLE['marker_size_small'],
                                    'opacity': 0.7
                                },
                                hovertemplate=f'{case}<br>FLL: %{{x:.4f}}<br>FSL: %{{y:.4f}}<extra></extra>'
                            ),
                            row=1, col=1
                        )
        
        # Plot 2: Delta analysis
        if 'delta_fll' in self.df.columns and 'delta_fsl' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['delta_fll'],
                    mode='lines',
                    name='Delta FLL',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width_thin']},
                    hovertemplate='Δ FLL: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['delta_fsl'],
                    mode='lines',
                    name='Delta FSL',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width_thin']},
                    hovertemplate='Δ FSL: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Correlation case distribution
        if 'corr_case' in self.df.columns:
            case_counts = self.df['corr_case'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=case_counts.index,
                    y=case_counts.values,
                    name='Case Frequency',
                    marker={'color': [case_colors.get(case, COLORS['primary']) for case in case_counts.index]},
                    hovertemplate='%{x}: %{y} occurrences<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Liquidation balance
        if 'liquidation_balance' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['liquidation_balance'],
                    mode='lines',
                    name='Liquidation Balance',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['primary'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Balance: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add reference line
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], 
                         row=2, col=2)
        
        fig.update_layout(
            **self.layout_theme,
            title='Liquidation Correlation Analysis',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_extreme_conditions_plot(self) -> go.Figure:
        """Create extreme conditions analysis visualization"""
        
        if 'is_rpn_extreme' not in self.df.columns:
            return self._create_empty_figure("No extreme conditions data available")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                'Price with Extreme RPN Conditions',
                'RPN Extreme Detection Over Time'
            ],
            vertical_spacing=0.1
        )
        
        # Plot 1: Price with extreme overlays
        if 'price' in self.df.columns:
            # Base price
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['price'],
                    mode='lines',
                    name='BTC Price',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Extreme conditions
            extreme_bull = self.df['is_rpn_extreme'] == 1
            extreme_bear = self.df['is_rpn_extreme'] == -1
            
            if extreme_bull.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][extreme_bull],
                        y=self.df['price'][extreme_bull],
                        mode='markers',
                        name='Extreme Bull',
                        marker={
                            'color': COLORS['success'],
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'star'
                        },
                        hovertemplate='Extreme Bull<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if extreme_bear.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][extreme_bear],
                        y=self.df['price'][extreme_bear],
                        mode='markers',
                        name='Extreme Bear',
                        marker={
                            'color': COLORS['danger'],
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'star'
                        },
                        hovertemplate='Extreme Bear<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Extreme detection timeline
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df['is_rpn_extreme'],
                mode='lines+markers',
                name='Extreme Indicator',
                line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width']},
                marker={'size': CHART_STYLE['marker_size_small']},
                hovertemplate='Extreme: %{y}<br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=1, line_dash="dash", line_color=COLORS['success'], row=2, col=1)
        fig.add_hline(y=-1, line_dash="dash", line_color=COLORS['danger'], row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], row=2, col=1)
        
        # Add extreme statistics if available
        if 'is_rpn_extreme_count' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['is_rpn_extreme_count'],
                    mode='lines',
                    name='Extreme Count (Weekly)',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width_thin']},
                    yaxis='y3',
                    hovertemplate='Weekly Extremes: %{y}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            **self.layout_theme,
            title='RPN Extreme Conditions Analysis',
            height=600,
            showlegend=True,
            yaxis3={'overlaying': 'y2', 'side': 'right', 'title': 'Extreme Count'}
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Extreme Indicator", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        return fig
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font={'size': 16, 'color': COLORS['text_muted']}
        )
        fig.update_layout(
            **self.layout_theme,
            height=400
        )
        return fig