"""
Binning Features Visualizations

Professional visualizations for binning and clustering features including
K-means analysis, momentum patterns, and bin distributions.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .config import COLORS, CHART_STYLE, LAYOUT_TEMPLATES, get_color_with_alpha, get_bin_color


class BinningVisualizer:
    """Professional visualizations for binning features"""
    
    def __init__(self, df: pd.DataFrame, time_col: str = 'time'):
        self.df = df.copy()
        self.time_col = time_col
        
        # Ensure time column is datetime
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            self.df = self.df.sort_values(time_col).reset_index(drop=True)
        
        self.layout_theme = LAYOUT_TEMPLATES['dark_financial']
    
    def create_price_color_plot(self) -> go.Figure:
        """Create price plot colored by bin_index - migrated from dashboard"""
        
        if 'price' not in self.df.columns or 'bin_index' not in self.df.columns:
            return self._create_empty_figure("Price or bin_index data not available")
        
        fig = go.Figure()
        
        # Define color mapping for bins (0-8)
        color_map = {
            0: COLORS['bin_colors'][0],  # Extreme bearish - Red
            1: COLORS['bin_colors'][1],
            2: COLORS['bin_colors'][2],
            3: COLORS['bin_colors'][3],
            4: COLORS['bin_colors'][4],  # Neutral - Yellow/Orange
            5: COLORS['bin_colors'][5],
            6: COLORS['bin_colors'][6],
            7: COLORS['bin_colors'][7],
            8: COLORS['bin_colors'][8]   # Extreme bullish - Green
        }
        
        # Add base price trace
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=self.df['price'],
            mode='lines',
            name='BTC Price',
            line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
            hovertemplate='Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add colored scatter points for each bin
        for bin_idx in range(9):
            if bin_idx in self.df['bin_index'].values:
                bin_data = self.df[self.df['bin_index'] == bin_idx]
                if not bin_data.empty:
                    fig.add_trace(go.Scatter(
                        x=bin_data[self.time_col],
                        y=bin_data['price'],
                        mode='markers',
                        name=f'Bin {bin_idx}',
                        marker={
                            'color': color_map.get(bin_idx, COLORS['text_muted']),
                            'size': CHART_STYLE['marker_size_small'],
                            'opacity': 0.8
                        },
                        hovertemplate=f'Bin {bin_idx}<br>Time: %{{x}}<br>Price: $%{{y:,.2f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            **self.layout_theme,
            title={
                'text': 'BTC Price Colored by Risk Priority Number Bins',
                'font': {'size': CHART_STYLE['title_size']},
                'x': 0.5
            },
            xaxis_title='Time',
            yaxis_title='Price ($)',
            height=CHART_STYLE['default_height'],
            showlegend=True,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02}
        )
        
        return fig
    
    def create_bin_analysis_dashboard(self) -> go.Figure:
        """Create comprehensive bin analysis dashboard"""
        
        if 'bin_index' not in self.df.columns:
            return self._create_empty_figure("Bin index data not available")
        
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            subplot_titles=[
                'Bin Index Over Time',
                'Bin Distribution',
                'Bin Momentum Analysis',
                'Bin Transition Matrix',
                'Market Regime Analysis',
                'Bin Statistics'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Bin index timeline with color coding
        colors = [get_bin_color(int(bin_val)) for bin_val in self.df['bin_index']]
        
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df['bin_index'],
                mode='lines+markers',
                name='Bin Index',
                line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                marker={'color': colors, 'size': CHART_STYLE['marker_size_small']},
                hovertemplate='Bin: %{y}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add bin level reference lines
        for i in range(9):
            fig.add_hline(y=i, line_dash="dot", line_color=get_bin_color(i), 
                         opacity=0.3, row=1, col=1)
        
        # Plot 2: Bin distribution histogram
        bin_counts = self.df['bin_index'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=bin_counts.index,
                y=bin_counts.values,
                name='Bin Frequency',
                marker={'color': [get_bin_color(i) for i in bin_counts.index]},
                hovertemplate='Bin %{x}: %{y} occurrences<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Plot 3: Bin momentum analysis
        if 'bin_momentum' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bin_momentum'],
                    mode='lines',
                    name='Bin Momentum',
                    line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Momentum: %{y}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add momentum direction if available
            if 'bin_mm_direction' in self.df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['bin_mm_direction'],
                        mode='lines',
                        name='Direction',
                        line={'color': COLORS['info'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y4',
                        hovertemplate='Direction: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Bin transition heatmap (simplified)
        if len(self.df) > 1:
            transitions = pd.crosstab(
                self.df['bin_index'].iloc[:-1], 
                self.df['bin_index'].iloc[1:], 
                normalize='index'
            ).fillna(0)
            
            fig.add_trace(
                go.Heatmap(
                    z=transitions.values,
                    x=transitions.columns,
                    y=transitions.index,
                    colorscale='Viridis',
                    name='Transition Probability',
                    hovertemplate='From %{y} to %{x}<br>Probability: %{z:.2%}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Plot 5: Market regime classification
        if 'bin_regime' in self.df.columns:
            regime_counts = self.df['bin_regime'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=regime_counts.index,
                    values=regime_counts.values,
                    name='Market Regimes',
                    hole=0.3,
                    hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Plot 6: Bin statistics over time
        if 'bin_volatility' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bin_volatility'],
                    mode='lines',
                    name='Bin Volatility',
                    line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Volatility: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=3, col=2
            )
            
            if 'bin_persistence' in self.df.columns:
                # Add persistence on secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['bin_persistence'],
                        mode='lines',
                        name='Persistence',
                        line={'color': COLORS['success'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y6',
                        hovertemplate='Persistence: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            **self.layout_theme,
            title='Comprehensive Bin Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update specific y-axis properties
        fig.update_yaxes(title_text="Bin Index", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Momentum", row=2, col=1)
        fig.update_yaxes(title_text="From Bin", row=2, col=2)
        fig.update_yaxes(title_text="Volatility", row=3, col=2)
        
        # Secondary y-axes
        fig.update_layout(
            yaxis4={'overlaying': 'y3', 'side': 'right', 'title': 'Direction'},
            yaxis6={'overlaying': 'y5', 'side': 'right', 'title': 'Persistence'}
        )
        
        return fig
    
    def create_clustering_analysis_plot(self) -> go.Figure:
        """Create K-means clustering analysis visualization"""
        
        if 'bin_index' not in self.df.columns or 'risk_priority_number' not in self.df.columns:
            return self._create_empty_figure("Clustering data not available")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'RPN Distribution by Bin',
                'Bin Centers and Distances',
                'Clustering Confidence',
                'Temporal Clustering Patterns'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: RPN distribution by bin (violin plot simulation)
        unique_bins = sorted(self.df['bin_index'].unique())
        for bin_idx in unique_bins:
            bin_data = self.df[self.df['bin_index'] == bin_idx]['risk_priority_number']
            if len(bin_data) > 0:
                fig.add_trace(
                    go.Box(
                        y=bin_data,
                        name=f'Bin {bin_idx}',
                        marker_color=get_bin_color(bin_idx),
                        boxpoints='outliers',
                        hovertemplate=f'Bin {bin_idx}<br>RPN: %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Bin centers and distances
        if 'bin_centers' in self.df.columns and 'bin_distance' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['bin_centers'],
                    y=self.df['bin_distance'],
                    mode='markers',
                    name='Distance to Center',
                    marker={
                        'color': [get_bin_color(int(b)) for b in self.df['bin_index']],
                        'size': CHART_STYLE['marker_size_small'],
                        'opacity': 0.6
                    },
                    hovertemplate='Center: %{x:.4f}<br>Distance: %{y:.4f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Clustering confidence over time
        if 'bin_confidence' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bin_confidence'],
                    mode='lines',
                    name='Confidence',
                    line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                    fill='tozeroy',
                    fillcolor=get_color_with_alpha(COLORS['success'], CHART_STYLE['fill_alpha']),
                    hovertemplate='Confidence: %{y:.3f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add confidence threshold
            fig.add_hline(y=0.7, line_dash="dash", line_color=COLORS['warning'], 
                         row=2, col=1)
        
        # Plot 4: Temporal patterns
        if 'bin_changed' in self.df.columns:
            # Bin changes over time
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['bin_changed'].cumsum(),
                    mode='lines',
                    name='Cumulative Changes',
                    line={'color': COLORS['warning'], 'width': CHART_STYLE['line_width']},
                    hovertemplate='Total Changes: %{y}<br>%{x}<extra></extra>'
                ),
                row=2, col=2
            )
            
            if 'bin_stability' in self.df.columns:
                # Add stability on secondary axis
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df['bin_stability'],
                        mode='lines',
                        name='Stability',
                        line={'color': COLORS['info'], 'width': CHART_STYLE['line_width_thin']},
                        yaxis='y4',
                        hovertemplate='Stability: %{y:.3f}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            **self.layout_theme,
            title='K-Means Clustering Analysis',
            height=600,
            showlegend=True,
            yaxis4={'overlaying': 'y3', 'side': 'right', 'title': 'Stability'}
        )
        
        return fig
    
    def create_momentum_patterns_plot(self) -> go.Figure:
        """Create bin momentum patterns visualization"""
        
        momentum_cols = ['bin_momentum', 'bin_turn', 'bin_mm_direction']
        available_cols = [col for col in momentum_cols if col in self.df.columns]
        
        if not available_cols:
            return self._create_empty_figure("No momentum data available")
        
        fig = make_subplots(
            rows=len(available_cols) + 1, cols=1,
            shared_xaxes=True,
            subplot_titles=['Bin Index'] + [col.replace('_', ' ').title() for col in available_cols],
            vertical_spacing=0.05
        )
        
        # Plot 1: Bin index for reference
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df['bin_index'],
                mode='lines',
                name='Bin Index',
                line={'color': COLORS['primary'], 'width': CHART_STYLE['line_width']},
                hovertemplate='Bin: %{y}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot momentum components
        colors = [COLORS['warning'], COLORS['danger'], COLORS['info']]
        for i, (col, color) in enumerate(zip(available_cols, colors)):
            row_num = i + 2
            
            if col == 'bin_momentum':
                # Bar plot for momentum
                fig.add_trace(
                    go.Bar(
                        x=self.df[self.time_col],
                        y=self.df[col],
                        name='Momentum',
                        marker={'color': [get_trend_color(val) for val in self.df[col]]},
                        hovertemplate='Momentum: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=row_num, col=1
                )
            elif col == 'bin_turn':
                # Scatter plot for turns
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[col],
                        mode='markers',
                        name='Turns',
                        marker={
                            'color': color,
                            'size': self.df[col] * 2 + 4,  # Size based on turn count
                            'opacity': 0.7
                        },
                        hovertemplate='Turns: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=row_num, col=1
                )
            else:
                # Line plot for direction
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=self.df[col],
                        mode='lines+markers',
                        name='Direction',
                        line={'color': color, 'width': CHART_STYLE['line_width']},
                        marker={'size': CHART_STYLE['marker_size_small']},
                        hovertemplate='Direction: %{y}<br>%{x}<extra></extra>'
                    ),
                    row=row_num, col=1
                )
                
                # Add reference lines for direction
                fig.add_hline(y=1, line_dash="dash", line_color=COLORS['success'], 
                             opacity=0.5, row=row_num, col=1)
                fig.add_hline(y=-1, line_dash="dash", line_color=COLORS['danger'], 
                             opacity=0.5, row=row_num, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color=COLORS['text_muted'], 
                             row=row_num, col=1)
        
        # Add enhanced features if available
        if 'momentum_strength' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df['momentum_strength'],
                    mode='lines',
                    name='Momentum Strength',
                    line={'color': COLORS['tertiary'], 'width': CHART_STYLE['line_width_thin']},
                    opacity=0.7,
                    hovertemplate='Strength: %{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            **self.layout_theme,
            title='Bin Momentum Patterns Analysis',
            height=max(600, len(available_cols) * 150 + 200),
            showlegend=True
        )
        
        return fig
    
    def create_market_regime_plot(self) -> go.Figure:
        """Create market regime analysis based on bins"""
        
        if 'bin_index' not in self.df.columns:
            return self._create_empty_figure("Bin data not available")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Market Regime Over Time',
                'Regime Distribution',
                'Extreme Conditions Analysis',
                'Regime Transitions'
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Define regimes based on bins
        def classify_regime(bin_val):
            if bin_val <= 1:
                return 'Extreme_Bull'
            elif bin_val <= 3:
                return 'Bull'
            elif bin_val <= 5:
                return 'Neutral'
            elif bin_val <= 7:
                return 'Bear'
            else:
                return 'Extreme_Bear'
        
        if 'bin_regime' not in self.df.columns:
            self.df['bin_regime'] = self.df['bin_index'].apply(classify_regime)
        
        # Plot 1: Regime timeline
        regime_colors = {
            'Extreme_Bull': COLORS['success'],
            'Bull': get_color_with_alpha(COLORS['success'], 0.7),
            'Neutral': COLORS['warning'],
            'Bear': get_color_with_alpha(COLORS['danger'], 0.7),
            'Extreme_Bear': COLORS['danger']
        }
        
        # Create regime bands
        regimes = self.df['bin_regime'].unique()
        for regime in regimes:
            mask = self.df['bin_regime'] == regime
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
        
        # Plot 2: Regime distribution pie chart
        regime_counts = self.df['bin_regime'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                name='Regime Distribution',
                marker={'colors': [regime_colors.get(regime, COLORS['primary']) for regime in regime_counts.index]},
                hole=0.3,
                hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Plot 3: Extreme conditions
        extreme_conditions = (self.df['bin_index'] <= 1) | (self.df['bin_index'] >= 7)
        extreme_ratio = extreme_conditions.rolling(window=24).mean()
        
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=extreme_ratio,
                mode='lines',
                name='Extreme Ratio',
                line={'color': COLORS['danger'], 'width': CHART_STYLE['line_width']},
                fill='tozeroy',
                fillcolor=get_color_with_alpha(COLORS['danger'], CHART_STYLE['fill_alpha']),
                hovertemplate='Extreme Ratio: %{y:.2%}<br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add market balance on secondary axis
        market_balance = 4 - np.abs(self.df['bin_index'] - 4)  # Distance from center
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=market_balance,
                mode='lines',
                name='Market Balance',
                line={'color': COLORS['info'], 'width': CHART_STYLE['line_width_thin']},
                yaxis='y4',
                hovertemplate='Balance: %{y}<br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 4: Regime persistence
        regime_changes = (self.df['bin_regime'] != self.df['bin_regime'].shift(1)).astype(int)
        regime_stability = 1 - regime_changes.rolling(window=24).mean()
        
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=regime_stability,
                mode='lines',
                name='Regime Stability',
                line={'color': COLORS['success'], 'width': CHART_STYLE['line_width']},
                hovertemplate='Stability: %{y:.2%}<br>%{x}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            **self.layout_theme,
            title='Market Regime Analysis',
            height=600,
            showlegend=True,
            yaxis4={'overlaying': 'y3', 'side': 'right', 'title': 'Market Balance'}
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