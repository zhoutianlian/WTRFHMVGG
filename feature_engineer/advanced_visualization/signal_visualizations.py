"""
Signal Features Visualizations

Professional visualizations for trading signal features including
signal analysis, performance tracking, and signal intelligence.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .config import COLORS, CHART_STYLE, LAYOUT_TEMPLATES, get_color_with_alpha, get_trend_color


class SignalVisualizer:
    """Professional visualizations for signal features"""
    
    def __init__(self, df: pd.DataFrame, time_col: str = 'time'):
        self.df = df.copy()
        self.time_col = time_col
        
        # Ensure time column is datetime
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            self.df = self.df.sort_values(time_col).reset_index(drop=True)
        
        self.layout_theme = LAYOUT_TEMPLATES['dark_financial']
    
    def create_signals_plot(self) -> go.Figure:
        """Create comprehensive trading signals plot - migrated from dashboard"""
        
        if 'signal_class' not in self.df.columns:
            return self._create_empty_figure("Signal data not available")
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=[
                'Price with Trading Signals',
                'Signal Levels (L1, L2, L3)',
                'Signal Strength & Quality',
                'Signal Performance Metrics'
            ]
        )
        
        # Plot 1: Price with signal overlays
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
            
            # Signal overlays
            signal_colors = {
                'Strong_Buy': COLORS['success'],
                'Buy': get_color_with_alpha(COLORS['success'], 0.8),
                'Weak_Buy': get_color_with_alpha(COLORS['success'], 0.6),
                'Strong_Sell': COLORS['danger'],
                'Sell': get_color_with_alpha(COLORS['danger'], 0.8),
                'Weak_Sell': get_color_with_alpha(COLORS['danger'], 0.6),
                'Neutral': COLORS['text_muted']
            }
            
            signal_symbols = {
                'Strong_Buy': 'triangle-up',
                'Buy': 'triangle-up',
                'Weak_Buy': 'triangle-up',
                'Strong_Sell': 'triangle-down',
                'Sell': 'triangle-down',
                'Weak_Sell': 'triangle-down',
                'Neutral': 'circle'
            }
            
            signal_sizes = {
                'Strong_Buy': CHART_STYLE['marker_size_large'],
                'Buy': CHART_STYLE['marker_size'],
                'Weak_Buy': CHART_STYLE['marker_size_small'],
                'Strong_Sell': CHART_STYLE['marker_size_large'],
                'Sell': CHART_STYLE['marker_size'],
                'Weak_Sell': CHART_STYLE['marker_size_small'],
                'Neutral': CHART_STYLE['marker_size_small']
            }
            
            for signal_type in signal_colors.keys():
                if signal_type == 'Neutral':
                    continue  # Skip neutral signals for clarity
                    
                mask = self.df['signal_class'] == signal_type
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=self.df[self.time_col][mask],
                            y=self.df['price'][mask],
                            mode='markers',
                            name=signal_type,
                            marker={
                                'color': signal_colors[signal_type],
                                'size': signal_sizes[signal_type],
                                'symbol': signal_symbols[signal_type],
                                'line': {'width': 1, 'color': 'white'}
                            },
                            hovertemplate=f'{signal_type}<br>Price: $%{{y:,.2f}}<br>%{{x}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # Plot 2: Signal levels
        signal_levels = ['signal_l1', 'signal_l2', 'signal_l3']
        level_colors = [COLORS['info'], COLORS['warning'], COLORS['danger']]
        
        for i, (level, color) in enumerate(zip(signal_levels, level_colors)):
            if level in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[level],
                        mode='lines+markers',
                        name=f'Level {i+1}',
                        line={'color': color, 'width': CHART_STYLE['line_width_thin']},
                        marker={'size': CHART_STYLE['marker_size_small']},
                        hovertemplate=f'{level}: %{{y}}<br>%{{x}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Add reference lines for signal levels
        for y_val in [-1, 0, 1]:
            line_color = COLORS['danger'] if y_val == -1 else (COLORS['success'] if y_val == 1 else COLORS['text_muted'])
            line_dash = "dot" if y_val == 0 else "dash"
            fig.add_hline(y=y_val, line_dash=line_dash, line_color=line_color, opacity=0.5, row=2, col=1)
        
        # Plot 3: Signal strength and quality
        if 'signal_strength' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['signal_strength'],
                    mode='lines',
                    name='Signal Strength',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['tertiary'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Strength: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
            
            if 'signal_quality' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['signal_quality'],
                        mode='lines',
                        name='Signal Quality',
                        line={'color': COLORS['secondary'], 'width': CHART_STYLE['line_width_thin'], 'dash': 'dash'},
                        hovertemplate='Quality: %{y:.3f}<br>%{x}<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Plot 4: Performance metrics
        performance_cols = ['signal_accuracy_1h', 'signal_accuracy_6h']
        perf_colors = [COLORS['success'], COLORS['info']]
        
        for i, (col, color) in enumerate(zip(performance_cols, perf_colors)):
            if col in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        line={'color': color, 'width': CHART_STYLE['line_width_thin']},
                        hovertemplate=f'{col}: %{{y:.2%}}<br>%{{x}}<extra></extra>'
                    ),
                    row=4, col=1
                )
        
        # Add 50% reference line for accuracy
        fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS['text_muted'], row=4, col=1)
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title={
                'text': 'Comprehensive Trading Signals Analysis',
                'font': {'size': CHART_STYLE['title_size']},
                'x': 0.5
            },
            height=CHART_STYLE['dashboard_height'],
            showlegend=True,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02}
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Signal Level", row=2, col=1)
        fig.update_yaxes(title_text="Strength/Quality", row=3, col=1)
        fig.update_yaxes(title_text="Accuracy", row=4, col=1)
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        return fig
    
    def create_signal_analysis_dashboard(self) -> go.Figure:
        """Create comprehensive signal analysis dashboard"""
        
        if 'signal_class' not in self.df.columns:
            return self._create_empty_figure("Signal data not available")
        
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Signal Distribution',
                'Signal Timing Analysis',
                'Signal Consistency Patterns',
                'Risk Assessment',
                'Signal Activity Clustering',
                'Performance by Signal Type'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Signal distribution pie chart
        signal_counts = self.df['signal_class'].value_counts()
        signal_colors_map = {
            'Strong_Buy': COLORS['success'],
            'Buy': get_color_with_alpha(COLORS['success'], 0.8),
            'Weak_Buy': get_color_with_alpha(COLORS['success'], 0.6),
            'Strong_Sell': COLORS['danger'],
            'Sell': get_color_with_alpha(COLORS['danger'], 0.8),
            'Weak_Sell': get_color_with_alpha(COLORS['danger'], 0.6),
            'Neutral': COLORS['text_muted']
        }
        
        fig.add_trace(
            go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                name='Signal Distribution',
                marker={'colors': [signal_colors_map.get(sig, COLORS['primary']) for sig in signal_counts.index]},
                hole=0.3,
                hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Signal timing analysis
        if 'signal_timing_score' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['signal_timing_score'],
                    mode='lines',
                    name='Timing Score',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Timing Score: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            if 'signal_persistence' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['signal_persistence'],
                        mode='lines',
                        name='Persistence',
                        line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y4',
                        hovertemplate='Persistence: %{y:.2%}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Signal consistency
        if 'signal_consistency' in self.df.columns:
            # Color by consistency value
            consistency_colors = []
            for val in self.df['signal_consistency']:
                if val > 1:
                    consistency_colors.append(COLORS['success'])
                elif val < -1:
                    consistency_colors.append(COLORS['danger'])
                else:
                    consistency_colors.append(COLORS['warning'])
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['signal_consistency'],
                    mode='markers',
                    name='Signal Consistency',
                    marker={
                        'color': consistency_colors,
                        'size': CHART_STYLE['marker_size_small'],
                        'opacity': 0.8
                    },
                    hovertemplate='Consistency: %{y}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add reference lines
            for y_val in [-3, -1, 1, 3]:
                line_color = COLORS['success'] if y_val > 0 else COLORS['danger']
                fig.add_hline(y=y_val, line_dash="dash", line_color=line_color, opacity=0.3, row=2, col=1)
        
        # Plot 4: Risk assessment
        if 'signal_risk' in self.df.columns:
            risk_counts = self.df['signal_risk'].value_counts()
            risk_colors = {
                'Low_Risk': COLORS['success'],
                'Medium_Risk': COLORS['warning'],
                'High_Risk': get_color_with_alpha(COLORS['danger'], 0.8),
                'Very_High_Risk': COLORS['danger']
            }
            
            fig.add_trace(
                go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    name='Risk Distribution',
                    marker={'color': [risk_colors.get(risk, COLORS['primary']) for risk in risk_counts.index]},
                    hovertemplate='%{x}: %{y} signals<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Plot 5: Signal activity clustering
        if 'signal_activity' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['signal_activity'],
                    mode='lines',
                    name='Signal Activity',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Activity: %{y}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
            
            if 'signal_cluster' in self.df.columns:
                # Highlight high activity clusters
                cluster_mask = self.df['signal_cluster'] == 1
                if cluster_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=self.df[self.time_col][cluster_mask],
                            y=self.df['signal_activity'][cluster_mask],
                            mode='markers',
                            name='High Activity Clusters',
                            marker={
                                'color': COLORS['danger'],
                                'size': CHART_STYLE['marker_size'],
                                'symbol': 'star'
                            },
                            hovertemplate='High Activity Cluster<br>Activity: %{y}<br>%{x}<extra></extra>'
                        ),
                        row=3, col=1
                    )
        
        # Plot 6: Performance by signal type
        if 'forward_return_1h' in self.df.columns:
            # Calculate average returns by signal type
            signal_performance = self.df.groupby('signal_class')['forward_return_1h'].mean().sort_values()
            
            perf_colors = []
            for ret in signal_performance.values:
                if ret > 0:
                    perf_colors.append(COLORS['success'])
                elif ret < 0:
                    perf_colors.append(COLORS['danger'])
                else:
                    perf_colors.append(COLORS['warning'])
            
            fig.add_trace(
                go.Bar(
                    x=signal_performance.index,
                    y=signal_performance.values,
                    name='Avg 1h Return',
                    marker={'color': perf_colors},
                    hovertemplate='%{x}<br>Avg Return: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=2
            )
            
            # Add zero reference line
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], row=3, col=2)
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title='Signal Analysis Dashboard',
            height=800,
            showlegend=True,
            yaxis4={'overlaying': 'y3', 'side': 'right', 'title': 'Persistence'}
        )
        
        return fig
    
    def create_ceiling_bottom_analysis_plot(self) -> go.Figure:
        """Create ceiling/bottom detection analysis"""
        
        if 'hit_ceiling_bottom' not in self.df.columns:
            return self._create_empty_figure("Ceiling/bottom data not available")
        
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Price with Ceiling/Bottom Hits',
                'Ceiling/Bottom Frequency',
                'Time Since Extremes',
                'Market Stress Analysis'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: Price with ceiling/bottom overlays
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
            
            # Ceiling hits
            ceiling_mask = self.df['hit_ceiling_bottom'] == -1
            if ceiling_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][ceiling_mask],
                        y=self.df['price'][ceiling_mask],
                        mode='markers',
                        name='Ceiling Hits',
                        marker={
                            'color': COLORS['danger'],
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'triangle-down'
                        },
                        hovertemplate='Ceiling Hit<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Bottom hits
            bottom_mask = self.df['hit_ceiling_bottom'] == 1
            if bottom_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][bottom_mask],
                        y=self.df['price'][bottom_mask],
                        mode='markers',
                        name='Bottom Hits',
                        marker={
                            'color': COLORS['success'],
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'triangle-up'
                        },
                        hovertemplate='Bottom Hit<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Frequency analysis
        if 'ceiling_frequency' in self.df.columns and 'bottom_frequency' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['ceiling_frequency'],
                    mode='lines',
                    name='Ceiling Frequency',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Ceiling Freq: %{y}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bottom_frequency'],
                    mode='lines',
                    name='Bottom Frequency',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Bottom Freq: %{y}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Time since extremes
        if 'time_since_ceiling' in self.df.columns and 'time_since_bottom' in self.df.columns:
            # Normalize times for better visualization
            max_time = 168  # 1 week
            ceiling_norm = np.minimum(self.df['time_since_ceiling'], max_time) / max_time
            bottom_norm = np.minimum(self.df['time_since_bottom'], max_time) / max_time
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=ceiling_norm,
                    mode='lines',
                    name='Time Since Ceiling',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['danger'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Time Since Ceiling: %{text}h<br>%{x}<extra></extra>',
                    text=self.df['time_since_ceiling']
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=bottom_norm,
                    mode='lines',
                    name='Time Since Bottom',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['success'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Time Since Bottom: %{text}h<br>%{x}<extra></extra>',
                    text=self.df['time_since_bottom']
                ),
                row=2, col=1
            )
        
        # Plot 4: Market stress indicators
        if 'extreme_market_stress' in self.df.columns:
            stress_ratio = self.df['extreme_market_stress'].rolling(window=24).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=stress_ratio,
                    mode='lines',
                    name='Market Stress',
                    line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Stress Ratio: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add stress threshold
            fig.add_hline(y=0.2, line_dash="dash", line_color=COLORS['danger'], row=2, col=2)
            
            if 'cb_strength' in self.df.columns:
                # Add ceiling/bottom strength
                strength_avg = self.df['cb_strength'].rolling(window=24).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=strength_avg,
                        mode='lines',
                        name='CB Strength',
                        line={'color': COLORS['info'], 'width': CHART_STYLE['line_width_thin'], 'dash': 'dash'},
                        hovertemplate='CB Strength: %{y:.3f}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            **self.layout_theme,
            title='Ceiling/Bottom Analysis',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_reversal_analysis_plot(self) -> go.Figure:
        """Create reversal signals analysis"""
        
        if 'reverse_ceiling_bottom' not in self.df.columns:
            return self._create_empty_figure("Reversal data not available")
        
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Price with Reversal Signals',
                'Beta Analysis for Reversals',
                'Reversal Confidence & Frequency',
                'Major Turning Points'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: Price with reversal signals
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
            
            # Bull reversals
            bull_reversal_mask = self.df['reverse_ceiling_bottom'] == 1
            if bull_reversal_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][bull_reversal_mask],
                        y=self.df['price'][bull_reversal_mask],
                        mode='markers',
                        name='Bull Reversals',
                        marker={
                            'color': COLORS['success'],
                            'size': CHART_STYLE['marker_size_large'],
                            'symbol': 'star',
                            'line': {'width': 2, 'color': 'white'}
                        },
                        hovertemplate='Bull Reversal<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Bear reversals
            bear_reversal_mask = self.df['reverse_ceiling_bottom'] == -1
            if bear_reversal_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][bear_reversal_mask],
                        y=self.df['price'][bear_reversal_mask],
                        mode='markers',
                        name='Bear Reversals',
                        marker={
                            'color': COLORS['danger'],
                            'size': CHART_STYLE['marker_size_large'],
                            'symbol': 'star',
                            'line': {'width': 2, 'color': 'white'}
                        },
                        hovertemplate='Bear Reversal<br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Beta analysis
        if 'diff_beta_8' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['diff_beta_8'],
                    mode='lines',
                    name='Beta (Slope)',
                    line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Beta: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add beta thresholds
            fig.add_hline(y=0.01, line_dash="dash", line_color=COLORS['success'], row=1, col=2)
            fig.add_hline(y=-0.01, line_dash="dash", line_color=COLORS['danger'], row=1, col=2)
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], row=1, col=2)
            
            # Highlight reversal periods
            reversal_mask = self.df['reverse_ceiling_bottom'] != 0
            if reversal_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][reversal_mask],
                        y=self.df['diff_beta_8'][reversal_mask],
                        mode='markers',
                        name='Reversal Beta',
                        marker={
                            'color': COLORS['tertiary'],
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'circle'
                        },
                        hovertemplate='Reversal Beta: %{y:.4f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Confidence and frequency
        if 'reversal_confidence' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['reversal_confidence'],
                    mode='lines',
                    name='Reversal Confidence',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['info'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Confidence: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            if 'reversal_frequency' in self.df.columns:
                freq_ma = self.df['reversal_frequency'].rolling(window=24).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=freq_ma,
                        mode='lines',
                        name='Reversal Frequency',
                        line={'color': COLORS['secondary'], 'width': CHART_STYLE['line_width_thin'], 'dash': 'dash'},
                        hovertemplate='Frequency: %{y:.1f}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Major turning points
        if 'major_turning_point' in self.df.columns:
            # Show major turning points as heatmap-style visualization
            turning_points = self.df['major_turning_point']
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=turning_points,
                    mode='markers',
                    name='Major Turning Points',
                    marker={
                        'color': turning_points,
                        'colorscale': [[0, COLORS['text_muted']], [1, COLORS['danger']]],
                        'size': turning_points * CHART_STYLE['marker_size'] + 4,
                        'opacity': 0.8
                    },
                    hovertemplate='Turning Point: %{y}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add reversal type distribution
            if 'reversal_type' in self.df.columns:
                # Count reversals by type over rolling window
                reversal_counts = self.df['reversal_type'].rolling(window=168).apply(
                    lambda x: (x != 'No_Reversal').sum(), raw=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=reversal_counts,
                        mode='lines',
                        name='Weekly Reversals',
                        line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width_thin']},
                        hovertemplate='Weekly Reversals: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            **self.layout_theme,
            title='Reversal Signals Analysis',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_liquidations_analysis_plot(self) -> go.Figure:
        """Create futures liquidations analysis plot - migrated from dashboard"""
        
        fll_col = 'fll_cwt_kf' if 'fll_cwt_kf' in self.df.columns else 'fll_normal'
        fsl_col = 'fsl_cwt_kf' if 'fsl_cwt_kf' in self.df.columns else 'fsl_normal'
        
        if fll_col not in self.df.columns or fsl_col not in self.df.columns:
            return self._create_empty_figure("Liquidation data not available")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=[
                'Futures Long vs Short Liquidations',
                'Liquidation Differential',
                'Liquidation Ratio & Balance'
            ]
        )
        
        # Plot 1: FLL vs FSL
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df[fll_col],
                mode='lines',
                name='Futures Long Liquidations',
                line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                hovertemplate='FLL: %{y:.4f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df[fsl_col],
                mode='lines',
                name='Futures Short Liquidations',
                line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                hovertemplate='FSL: %{y:.4f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Differential
        diff_col = 'diff_ls_cwt_kf' if 'diff_ls_cwt_kf' in self.df.columns else None
        if diff_col and diff_col in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df[diff_col],
                    mode='lines',
                    name='FLL - FSL Differential',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['tertiary'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Differential: %{y:.4f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero reference line
            fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], row=2, col=1)
        
        # Plot 3: Ratio and balance
        if 'fll_fsl_ratio' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['fll_fsl_ratio'],
                    mode='lines',
                    name='FLL/FSL Ratio',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Ratio: %{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Add ratio reference lines
            fig.add_hline(y=1, line_dash="dot", line_color=COLORS['text_muted'], row=3, col=1)
            fig.add_hline(y=2, line_dash="dash", line_color=COLORS['success'], opacity=0.5, row=3, col=1)
            fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS['danger'], opacity=0.5, row=3, col=1)
        elif fll_col in self.df.columns and fsl_col in self.df.columns:
            # Calculate ratio inline
            ratio = self.df[fll_col] / (self.df[fsl_col] + 1e-10)
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=ratio,
                    mode='lines',
                    name='FLL/FSL Ratio',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Ratio: %{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title={
                'text': 'Futures Liquidations Analysis',
                'font': {'size': CHART_STYLE['title_size']},
                'x': 0.5
            },
            height=CHART_STYLE['dashboard_height'],
            showlegend=True
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Liquidations", row=1, col=1)
        fig.update_yaxes(title_text="Differential", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
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