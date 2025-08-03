# Extracted PlotlyVisualizer class from backend.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict
import sys
import os

# Add feature_engineer to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_engineer'))


class PlotlyVisualizer:
    """Convert matplotlib-style plots to Plotly for interactive web display"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data['time'] = pd.to_datetime(self.data['time'])
        
    def create_price_color_plot(self) -> Dict:
        """Create price plot colored by bin_index"""
        fig = go.Figure()
        
        # Define color mapping
        color_map = {
            0: '#ffc0b0', 1: '#ffb3a0', 2: '#ffa590',
            3: '#ff9880', 4: '#ff8a70', 5: '#ff7d60',
            6: '#ff6f50', 7: '#ff6240', 8: '#ff5430'
        }
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=self.data['time'],
            y=self.data['price'],
            mode='lines',
            name='BTC Price',
            line=dict(color='#f1c232', width=2),
            hovertemplate='Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add colored scatter points for bins
        for bin_idx, color in color_map.items():
            bin_data = self.data[self.data['bin_index'] == bin_idx]
            if not bin_data.empty:
                fig.add_trace(go.Scatter(
                    x=bin_data['time'],
                    y=bin_data['price'],
                    mode='markers',
                    name=f'Bin {bin_idx}',
                    marker=dict(color=color, size=4),
                    hovertemplate=f'Bin {bin_idx}<br>Time: %{{x}}<br>Price: $%{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title='BTC Price Colored by Risk Priority Number Bins',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig.to_dict()
    
    def create_rpn_dominance_plot(self) -> Dict:
        """Create RPN and dominance plot"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('BTC Price with Dominance', 'Risk Priority Number')
        )
        
        # Price trace
        fig.add_trace(
            go.Scatter(
                x=self.data['time'],
                y=self.data['price'],
                name='BTC Price',
                line=dict(color='#f1c232', width=2),
                hovertemplate='Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add dominance background colors
        for i in range(1, len(self.data)):
            if self.data.iloc[i]['dominance'] == 1:
                fig.add_vrect(
                    x0=self.data.iloc[i-1]['time'],
                    x1=self.data.iloc[i]['time'],
                    fillcolor='#b6d7a8',
                    opacity=0.2,
                    layer='below',
                    line_width=0,
                    row=1, col=1
                )
            elif self.data.iloc[i]['dominance'] == -1:
                fig.add_vrect(
                    x0=self.data.iloc[i-1]['time'],
                    x1=self.data.iloc[i]['time'],
                    fillcolor='#ea9999',
                    opacity=0.2,
                    layer='below',
                    line_width=0,
                    row=1, col=1
                )
        
        # RPN trace
        fig.add_trace(
            go.Scatter(
                x=self.data['time'],
                y=self.data['risk_priority_number'],
                name='RPN',
                line=dict(color='#3d85c6', width=1),
                hovertemplate='RPN: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add RPN smooth line
        if 'lld_cwt_kf_smooth' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data['time'],
                    y=self.data['lld_cwt_kf_smooth'],
                    name='RPN Smooth',
                    line=dict(color='#f1c232', width=1),
                    hovertemplate='RPN Smooth: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='RPN', row=2, col=1)
        
        fig.update_layout(
            height=800,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig.to_dict()
    
    def create_fll_fsl_plot(self) -> Dict:
        """Create FLL/FSL liquidations plot"""
        fig = go.Figure()
        
        # Add FLL trace
        fig.add_trace(go.Scatter(
            x=self.data['time'],
            y=self.data['fll_cwt_kf'],
            name='Futures Long Liquidations',
            line=dict(color='#8fce00', width=1),
            hovertemplate='FLL: %{y:.4f}<extra></extra>'
        ))
        
        # Add FSL trace
        fig.add_trace(go.Scatter(
            x=self.data['time'],
            y=self.data['fsl_cwt_kf'],
            name='Futures Short Liquidations',
            line=dict(color='#f44336', width=1),
            hovertemplate='FSL: %{y:.4f}<extra></extra>'
        ))
        
        # Add correlation case coloring
        if 'corr_case' in self.data.columns:
            corr_colors = {
                'LUSUL': '#b6d7a8', 'LUSUS': '#93c47d',
                'LDSDL': '#ea9999', 'LDSDS': '#f4cccc',
                'LUSDL': '#ffe599', 'LUSDS': '#ffd966',
                'LDSUL': '#9fc5e8', 'LDSUS': '#cfe2f3'
            }
            
            for case, color in corr_colors.items():
                case_data = self.data[self.data['corr_case'] == case]
                if not case_data.empty:
                    fig.add_trace(go.Scatter(
                        x=case_data['time'],
                        y=case_data['fll_cwt_kf'],
                        mode='markers',
                        name=case,
                        marker=dict(color=color, size=3),
                        showlegend=True,
                        hovertemplate=f'{case}<br>Time: %{{x}}<br>FLL: %{{y:.4f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title='Futures Liquidations Analysis',
            xaxis_title='Time',
            yaxis_title='Liquidations',
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig.to_dict()
    
    def create_dominance_status_plot(self) -> Dict:
        """Create dominance status plot"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Dominance Duration', 'Dominance Time', 'Price with Dominance')
        )
        
        # Dominance duration
        fig.add_trace(
            go.Scatter(
                x=self.data['time'],
                y=self.data['dominance_duration_total'],
                name='Duration Total',
                fill='tozeroy',
                line=dict(color='#6aa84f', width=1),
                hovertemplate='Duration: %{y} hours<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Dominance time
        fig.add_trace(
            go.Scatter(
                x=self.data['time'],
                y=self.data['dominance_time'],
                name='Dominance Time',
                mode='markers',
                marker=dict(color='#f44336', size=3),
                hovertemplate='Dom Time: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Price with dominance class
        fig.add_trace(
            go.Scatter(
                x=self.data['time'],
                y=self.data['price'],
                name='BTC Price',
                line=dict(color='#f1c232', width=2),
                hovertemplate='Price: $%{y:,.2f}<br>Dom: %{customdata}<extra></extra>',
                customdata=self.data['dominance_class']
            ),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text='Time', row=3, col=1)
        fig.update_yaxes(title_text='Hours', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=2, col=1)
        fig.update_yaxes(title_text='Price ($)', row=3, col=1)
        
        fig.update_layout(
            height=900,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig.to_dict()
    
    def create_signals_plot(self) -> Dict:
        """Create signals plot with ceiling/bottom indicators"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=self.data['time'],
            y=self.data['price'],
            name='BTC Price',
            line=dict(color='#f1c232', width=2, opacity=0.6),
            hovertemplate='Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # Hit ceiling markers
        ceiling_data = self.data[self.data['hit_ceiling_bottom'] == -1]
        if not ceiling_data.empty:
            fig.add_trace(go.Scatter(
                x=ceiling_data['time'],
                y=ceiling_data['price'],
                mode='markers',
                name='Hit Ceiling',
                marker=dict(color='#cc0000', size=10, symbol='triangle-down'),
                hovertemplate='Hit Ceiling<br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        # Hit bottom markers
        bottom_data = self.data[self.data['hit_ceiling_bottom'] == 1]
        if not bottom_data.empty:
            fig.add_trace(go.Scatter(
                x=bottom_data['time'],
                y=bottom_data['price'],
                mode='markers',
                name='Hit Bottom',
                marker=dict(color='#38761d', size=10, symbol='triangle-up'),
                hovertemplate='Hit Bottom<br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        # Reverse ceiling/bottom markers
        rev_ceiling = self.data[self.data['reverse_ceiling_bottom'] == -1]
        if not rev_ceiling.empty:
            fig.add_trace(go.Scatter(
                x=rev_ceiling['time'],
                y=rev_ceiling['price'],
                mode='markers',
                name='Reverse Ceiling',
                marker=dict(color='#e06666', size=8, symbol='square'),
                hovertemplate='Reverse Ceiling<br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        rev_bottom = self.data[self.data['reverse_ceiling_bottom'] == 1]
        if not rev_bottom.empty:
            fig.add_trace(go.Scatter(
                x=rev_bottom['time'],
                y=rev_bottom['price'],
                mode='markers',
                name='Reverse Bottom',
                marker=dict(color='#6aa84f', size=8, symbol='square'),
                hovertemplate='Reverse Bottom<br>Time: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Trading Signals',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            hovermode='closest',
            template='plotly_dark'
        )
        
        return fig.to_dict()