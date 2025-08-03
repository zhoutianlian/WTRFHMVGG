"""
Dominance Features Visualizations

Professional visualizations for market dominance features including
dominance analysis, regime detection, and status tracking.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .config import COLORS, CHART_STYLE, LAYOUT_TEMPLATES, get_color_with_alpha, get_trend_color


class DominanceVisualizer:
    """Professional visualizations for dominance features"""
    
    def __init__(self, df: pd.DataFrame, time_col: str = 'time'):
        self.df = df.copy()
        self.time_col = time_col
        
        # Ensure time column is datetime
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            self.df = self.df.sort_values(time_col).reset_index(drop=True)
        
        self.layout_theme = LAYOUT_TEMPLATES['dark_financial']
    
    def create_dominance_status_plot(self) -> go.Figure:
        """Create dominance status analysis plot - migrated from dashboard"""
        
        if 'dominance' not in self.df.columns:
            return self._create_empty_figure("Dominance data not available")
        
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Dominance Timeline',
                'Dominance Class Distribution',
                'Dominance Duration Analysis',
                'Regime Transitions',
                'Market Efficiency Metrics',
                'Dominance Confidence'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Plot 1: Dominance timeline with colored regions
        dominance_colors = {
            1: COLORS['success'],    # Bull dominance
            0: COLORS['warning'],    # Congestion
            -1: COLORS['danger']     # Bear dominance
        }
        
        for dom_val, color in dominance_colors.items():
            mask = self.df['dominance'] == dom_val
            if mask.any():
                dom_name = {1: 'Bull', 0: 'Congestion', -1: 'Bear'}[dom_val]
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][mask],
                        y=self.df['dominance'][mask],
                        mode='markers',
                        name=f'{dom_name} Dominance',
                        marker={
                            'color': color,
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'square'
                        },
                        hovertemplate=f'{dom_name} Dominance<br>%{{x}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Add dominance line
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df['dominance'],
                mode='lines',
                name='Dominance Trend',
                line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width_thin']},
                opacity=0.7,
                hovertemplate='Dominance: %{y}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=1, line_dash="dash", line_color=COLORS['success'], opacity=0.5, row=1, col=1)
        fig.add_hline(y=-1, line_dash="dash", line_color=COLORS['danger'], opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS['warning'], row=1, col=1)
        
        # Plot 2: Dominance class distribution
        if 'dominance_class' in self.df.columns:
            class_counts = self.df['dominance_class'].value_counts()
            class_colors = {
                'Bull': COLORS['success'],
                'Bear': COLORS['danger'],
                'Congestion': COLORS['warning']
            }
            
            fig.add_trace(
                go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    name='Dominance Classes',
                    marker={'colors': [class_colors.get(cls, COLORS['primary']) for cls in class_counts.index]},
                    hole=0.3,
                    hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Duration analysis
        if 'dominance_duration' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['dominance_duration'],
                    mode='lines',
                    name='Duration',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Duration: %{y}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add average duration on secondary axis
            avg_duration = self.df['dominance_duration'].rolling(window=24).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=avg_duration,
                    mode='lines',
                    name='Avg Duration',
                    line={'color': COLORS['secondary'], 'width': CHART_STYLE['line_width_thin'], 'dash': 'dash'},
                    yaxis='y4',
                    hovertemplate='Avg Duration: %{y:.1f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Regime transitions
        if 'dominance_transition_type' in self.df.columns:
            transition_counts = self.df['dominance_transition_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=transition_counts.index,
                    y=transition_counts.values,
                    name='Transition Types',
                    marker={'color': COLORS['tertiary']},
                    hovertemplate='%{x}: %{y} transitions<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Plot 5: Market efficiency metrics
        if 'regime_stability' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['regime_stability'],
                    mode='lines',
                    name='Regime Stability',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Stability: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
            
            if 'transition_frequency' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['transition_frequency'],
                        mode='lines',
                        name='Transition Frequency',
                        line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y6',
                        hovertemplate='Transitions: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Plot 6: Dominance confidence
        if 'dominance_confidence' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['dominance_confidence'],
                    mode='lines',
                    name='Confidence',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['primary'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Confidence: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=3, col=2
            )
            
            # Add confidence threshold
            fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS['success'], row=3, col=2)
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title={
                'text': 'Market Dominance Status Analysis',
                'font': {'size': CHART_STYLE['title_size']},
                'x': 0.5
            },
            height=800,
            showlegend=True,
            yaxis4={'overlaying': 'y3', 'side': 'right', 'title': 'Avg Duration'},
            yaxis6={'overlaying': 'y5', 'side': 'right', 'title': 'Transitions'}
        )
        
        return fig
    
    def create_market_regime_dashboard(self) -> go.Figure:
        """Create comprehensive market regime analysis dashboard"""
        
        if 'market_regime' not in self.df.columns:
            return self._create_empty_figure("Market regime data not available")
        
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Market Regime Timeline',
                'Regime Strength Analysis',
                'Bull vs Bear Time Ratios',
                'Regime Conviction',
                'Market Efficiency',
                'Trend Consistency'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Plot 1: Market regime timeline
        regime_colors = {
            'Strong_Bull': COLORS['success'],
            'Weak_Bull': get_color_with_alpha(COLORS['success'], 0.7),
            'Congestion': COLORS['warning'],
            'Weak_Bear': get_color_with_alpha(COLORS['danger'], 0.7),
            'Strong_Bear': COLORS['danger'],
            'Mixed': COLORS['text_muted']
        }
        
        regimes = self.df['market_regime'].unique()
        for regime in regimes:
            mask = self.df['market_regime'] == regime
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col][mask],
                        y=[regime] * mask.sum(),
                        mode='markers',
                        name=regime,
                        marker={
                            'color': regime_colors.get(regime, COLORS['primary']),
                            'size': CHART_STYLE['marker_size'],
                            'symbol': 'square'
                        },
                        hovertemplate=f'{regime}<br>%{{x}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Regime strength with volatility
        if 'regime_strength' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['regime_strength'],
                    mode='lines',
                    name='Regime Strength',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Strength: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            if 'regime_volatility' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['regime_volatility'],
                        mode='lines',
                        name='Regime Volatility',
                        line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y4',
                        hovertemplate='Volatility: %{y:.3f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Bull vs Bear time ratios
        if 'bull_time_ratio' in self.df.columns and 'bear_time_ratio' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bull_time_ratio'],
                    mode='lines',
                    name='Bull Time Ratio',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['success'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Bull Ratio: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bear_time_ratio'],
                    mode='lines',
                    name='Bear Time Ratio',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['danger'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Bear Ratio: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add balance line
            fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS['text_muted'], row=2, col=1)
        
        # Plot 4: Regime conviction
        if 'regime_conviction' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['regime_conviction'],
                    mode='lines',
                    name='Regime Conviction',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Conviction: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add conviction levels
            fig.add_hline(y=0.3, line_dash="dash", line_color=COLORS['warning'], 
                         opacity=0.5, row=2, col=2)
            fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS['success'], 
                         opacity=0.5, row=2, col=2)
        
        # Plot 5: Market efficiency
        if 'market_efficiency' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['market_efficiency'],
                    mode='lines',
                    name='Market Efficiency',
                    line={'color': COLORS['info'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Efficiency: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=3, col=1
            )
            
            if 'regime_change_probability' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['regime_change_probability'],
                        mode='lines',
                        name='Change Probability',
                        line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y6',
                        hovertemplate='Change Prob: %{y:.2%}<br>%{x}<extra></extra>'
                    ),
                    row=3, col=1
                )
        
        # Plot 6: Trend consistency
        if 'trend_consistency' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['trend_consistency'],
                    mode='lines',
                    name='Trend Consistency',
                    line={'color': COLORS['secondary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Consistency: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title='Market Regime Analysis Dashboard',
            height=800,
            showlegend=True,
            yaxis4={'overlaying': 'y3', 'side': 'right', 'title': 'Volatility'},
            yaxis6={'overlaying': 'y5', 'side': 'right', 'title': 'Change Prob'}
        )
        
        return fig
    
    def create_dominance_patterns_plot(self) -> go.Figure:
        """Create dominance patterns and persistence analysis"""
        
        if 'dominance' not in self.df.columns:
            return self._create_empty_figure("Dominance data not available")
        
        fig = make_subplots(
            rows=2, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Dominance Persistence Patterns',
                'Dominance Quality Metrics',
                'Regime Change Analysis',
                'Market Stress Indicators'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: Dominance persistence
        if 'dominance_persistence_score' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['dominance_persistence_score'],
                    mode='lines',
                    name='Persistence Score',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['success'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Persistence: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            if 'dominance_streak' in self.df.columns:
                # Add streak length as scatter
                streaks = self.df['dominance_streak']
                significant_streaks = streaks[streaks > 12]  # Streaks longer than 12 periods
                
                if len(significant_streaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=self.df[self.time_col][streaks > 12],
                            y=self.df['dominance_persistence_score'][streaks > 12],
                            mode='markers',
                            name='Long Streaks',
                            marker={
                                'color': COLORS['warning'],
                                'size': np.minimum(significant_streaks * 0.5 + 4, 20),  # Size based on streak length
                                'opacity': 0.7
                            },
                            hovertemplate='Streak: %{text}<br>Persistence: %{y:.2%}<br>%{x}<extra></extra>',
                            text=significant_streaks
                        ),
                        row=1, col=1
                    )
        
        # Plot 2: Dominance quality
        if 'dominance_quality' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['dominance_quality'],
                    mode='lines',
                    name='Dominance Quality',
                    line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Quality: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            if 'dominance_trend_strength' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['dominance_trend_strength'],
                        mode='lines',
                        name='Trend Strength',
                        line={'color': COLORS['info'], 'width': CHART_STYLE['line_width_thin'], 'dash': 'dash'},
                        hovertemplate='Trend Strength: %{y:.3f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Regime change analysis
        if 'regime_change_probability' in self.df.columns:
            # Color-coded by probability
            prob_colors = []
            for prob in self.df['regime_change_probability']:
                if prob > 0.7:
                    prob_colors.append(COLORS['danger'])
                elif prob > 0.4:
                    prob_colors.append(COLORS['warning'])
                else:
                    prob_colors.append(COLORS['success'])
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['regime_change_probability'],
                    mode='markers',
                    name='Change Probability',
                    marker={
                        'color': prob_colors,
                        'size': CHART_STYLE['marker_size_small'],
                        'opacity': 0.8
                    },
                    hovertemplate='Change Prob: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add probability thresholds
            fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS['danger'], row=2, col=1)
            fig.add_hline(y=0.4, line_dash="dash", line_color=COLORS['warning'], row=2, col=1)
        
        # Plot 4: Market stress indicators
        if 'extreme_conditions' in self.df.columns:
            extreme_ratio = self.df['extreme_conditions'].rolling(window=24).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=extreme_ratio,
                    mode='lines',
                    name='Extreme Conditions',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Extreme Ratio: %{y:.2%}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            if 'market_imbalance' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['market_imbalance'],
                        mode='lines',
                        name='Market Imbalance',
                        line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width_thin']},
                        hovertemplate='Imbalance: %{y:.3f}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            **self.layout_theme,
            title='Dominance Patterns & Market Stress Analysis',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_dominance_correlation_plot(self) -> go.Figure:
        """Create dominance correlation with other market factors"""
        
        if 'dominance' not in self.df.columns:
            return self._create_empty_figure("Dominance data not available")
        
        # Find relevant columns for correlation
        relevant_cols = []
        potential_cols = [
            'bin_index', 'risk_priority_number', 'price', 'bin_volatility',
            'fll_cwt_kf', 'fsl_cwt_kf', 'diff_ls_cwt_kf'
        ]
        
        for col in potential_cols:
            if col in self.df.columns:
                relevant_cols.append(col)
        
        if len(relevant_cols) < 2:
            return self._create_empty_figure("Insufficient data for correlation analysis")
        
        # Calculate correlations with dominance
        correlations = {}
        for col in relevant_cols:
            corr = self.df['dominance'].corr(self.df[col])
            if not pd.isna(corr):
                correlations[col] = corr
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Dominance vs Key Metrics',
                'Rolling Correlations',
                'Correlation Strength',
                'Dominance Scatter Analysis'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: Static correlations bar chart
        if correlations:
            sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            cols, corr_vals = zip(*sorted_corrs)
            
            colors = [get_trend_color(val) for val in corr_vals]
            
            fig.add_trace(
                go.Bar(
                    x=list(cols),
                    y=list(corr_vals),
                    name='Correlations',
                    marker={'color': colors},
                    hovertemplate='%{x}<br>Correlation: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Rolling correlations for top metrics
        top_metrics = list(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))[:3]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['warning']]
        
        for i, (metric, _) in enumerate(top_metrics):
            rolling_corr = self.df['dominance'].rolling(window=168).corr(self.df[metric])
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=rolling_corr,
                    mode='lines',
                    name=f'{metric} Correlation',
                    line={'color': colors[i], 'width': CHART_STYLE['line_width_thin']},
                    hovertemplate=f'{metric} Corr: %{{y:.3f}}<br>%{{x}}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Add zero reference line
        fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], row=1, col=2)
        
        # Plot 3: Correlation strength over time
        if len(top_metrics) > 0:
            # Calculate combined correlation strength
            combined_strength = 0
            for metric, _ in top_metrics:
                rolling_corr = self.df['dominance'].rolling(window=168).corr(self.df[metric])
                combined_strength += np.abs(rolling_corr)
            
            combined_strength /= len(top_metrics)
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=combined_strength,
                    mode='lines',
                    name='Correlation Strength',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['tertiary'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Correlation Strength: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Scatter plot of dominance vs strongest correlated metric
        if top_metrics:
            strongest_metric = top_metrics[0][0]
            
            # Color by time (gradient)
            time_colors = np.arange(len(self.df))
            
            fig.add_trace(
                go.Scatter(
                    x=self.df[strongest_metric],
                    y=self.df['dominance'],
                    mode='markers',
                    name=f'Dom vs {strongest_metric}',
                    marker={
                        'color': time_colors,
                        'colorscale': 'Viridis',
                        'size': CHART_STYLE['marker_size_small'],
                        'opacity': 0.7,
                        'colorbar': {'title': 'Time', 'x': 1.1}
                    },
                    hovertemplate=f'{strongest_metric}: %{{x:.3f}}<br>Dominance: %{{y}}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            **self.layout_theme,
            title='Dominance Correlation Analysis',
            height=600,
            showlegend=True
        )
        
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