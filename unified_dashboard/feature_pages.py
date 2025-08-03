"""
Feature-Specific Pages

Individual page implementations for each feature category.
Each page provides comprehensive analysis and visualization for its specific features.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from .components import (
    PageHeader, ChartContainer, MetricCard, StatusIndicator,
    TimeRangeSelector, ComponentCallbacks
)
from .visualization_engine import VisualizationEngine


class BasePage:
    """Base class for all feature pages"""
    
    def __init__(self, engine: VisualizationEngine):
        self.engine = engine
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_layout(self) -> html.Div:
        """Create the page layout - to be implemented by subclasses"""
        raise NotImplementedError
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup page-specific callbacks - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_page_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get data for this page's feature category"""
        return self.engine.get_data(start_time, end_time, [self.category])
    
    def create_metrics_row(self, df: pd.DataFrame) -> html.Div:
        """Create metrics summary row for the page"""
        if df.empty:
            return html.Div("No data available", style={'color': 'white'})
        
        # Basic metrics that apply to all categories
        metrics = [
            MetricCard.create(
                "Data Points", 
                f"{len(df):,}", 
                f"From {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}",
                '#00D4FF', 
                '📊'
            ),
            MetricCard.create(
                "Time Span", 
                f"{(df['time'].max() - df['time'].min()).days} days",
                f"Latest: {df['time'].max().strftime('%H:%M')}",
                '#4ECDC4',
                '⏰'
            ),
            MetricCard.create(
                "Current Price", 
                f"${df['price'].iloc[-1]:,.2f}" if 'price' in df.columns else "N/A",
                f"Range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}" if 'price' in df.columns else "",
                '#10B981',
                '💰'
            )
        ]
        
        return html.Div(metrics, style={
            'display': 'flex',
            'gap': '20px',
            'marginBottom': '30px',
            'flexWrap': 'wrap'
        })


class RPNFeaturePage(BasePage):
    """Risk Priority Number Features Page"""
    
    category = 'rpn'
    
    def create_layout(self) -> html.Div:
        """Create RPN features page layout"""
        
        return html.Div([
            PageHeader.create(
                "🎯 Risk Priority Number (RPN) Features",
                "Advanced analysis of liquidation-based risk indicators with CWT detrending, Kalman filtering, and correlation analysis",
                6  # Number of RPN features
            ),
            
            TimeRangeSelector.create('rpn-time-range'),
            
            html.Div(id='rpn-metrics-row'),
            
            html.Div([
                ChartContainer.create('rpn-dominance-chart', '📈 RPN & Market Dominance Analysis', 600),
                ChartContainer.create('rpn-cwt-chart', '🌊 CWT Detrending Analysis', 500),
                ChartContainer.create('rpn-kalman-chart', '🎯 Kalman Filtering Analysis', 500),
                ChartContainer.create('rpn-correlation-chart', '🔗 Correlation Analysis', 500),
                ChartContainer.create('rpn-extreme-chart', '⚡ Extreme Conditions Detection', 400),
            ])
        ])
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup RPN page callbacks"""
        
        ComponentCallbacks.setup_time_range_callback(app, 'rpn-time-range')
        
        @app.callback(
            [Output('rpn-metrics-row', 'children'),
             Output('rpn-dominance-chart', 'figure'),
             Output('rpn-cwt-chart', 'figure'),
             Output('rpn-kalman-chart', 'figure'),
             Output('rpn-correlation-chart', 'figure'),
             Output('rpn-extreme-chart', 'figure')],
            [Input('rpn-time-range-refresh-btn', 'n_clicks')],
            [State('rpn-time-range-dropdown', 'value'),
             State('rpn-time-range-start-date', 'date'),
             State('rpn-time-range-end-date', 'date')]
        )
        def update_rpn_page(n_clicks, time_range, start_date, end_date):
            start_time, end_time = ComponentCallbacks.parse_time_range(time_range, start_date, end_date)
            
            try:
                # Get RPN data
                df = self.get_page_data(start_time, end_time)
                
                if df.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
                    empty_fig.update_layout(template='plotly_dark')
                    
                    return (
                        StatusIndicator.create('warning', 'No RPN data available for selected time range'),
                        empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
                    )
                
                # Create metrics row
                metrics_row = self.create_metrics_row(df)
                
                # Add RPN-specific metrics
                rpn_metrics = []
                if 'risk_priority_number' in df.columns:
                    rpn_metrics.append(
                        MetricCard.create(
                            "Average RPN",
                            f"{df['risk_priority_number'].mean():.4f}",
                            f"Range: {df['risk_priority_number'].min():.4f} - {df['risk_priority_number'].max():.4f}",
                            '#FF6B6B',
                            '🎯'
                        )
                    )
                
                if 'is_rpn_extreme' in df.columns:
                    extreme_count = df['is_rpn_extreme'].sum()
                    rpn_metrics.append(
                        MetricCard.create(
                            "Extreme Events",
                            f"{extreme_count:,}",
                            f"{extreme_count/len(df)*100:.1f}% of total data points",
                            '#F59E0B',
                            '⚡'
                        )
                    )
                
                if rpn_metrics:
                    metrics_row = html.Div([
                        metrics_row,
                        html.Div(rpn_metrics, style={
                            'display': 'flex',
                            'gap': '20px',
                            'marginTop': '20px',
                            'flexWrap': 'wrap'
                        })
                    ])
                
                # Create charts
                charts = []
                chart_types = ['dominance_plot', 'cwt_analysis', 'kalman_analysis', 'correlation_analysis', 'extreme_conditions']
                
                for chart_type in chart_types:
                    try:
                        fig = self.engine.create_chart(chart_type, 'rpn', df)
                        charts.append(fig)
                    except Exception as e:
                        self.logger.error(f"Error creating {chart_type} chart: {e}")
                        empty_fig = go.Figure()
                        empty_fig.add_annotation(text=f"Error loading {chart_type}", xref="paper", yref="paper", x=0.5, y=0.5)
                        empty_fig.update_layout(template='plotly_dark')
                        charts.append(empty_fig)
                
                return [metrics_row] + charts
                
            except Exception as e:
                self.logger.error(f"Error updating RPN page: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
                error_fig.update_layout(template='plotly_dark')
                
                return (
                    StatusIndicator.create('error', f'Error loading RPN data: {str(e)}'),
                    error_fig, error_fig, error_fig, error_fig, error_fig
                )


class BinningFeaturePage(BasePage):
    """Binning Features Page"""
    
    category = 'binning'
    
    def create_layout(self) -> html.Div:
        """Create binning features page layout"""
        
        return html.Div([
            PageHeader.create(
                "📦 Binning Features",
                "K-means clustering analysis and market regime classification based on liquidation patterns",
                4  # Number of binning features
            ),
            
            TimeRangeSelector.create('binning-time-range'),
            
            html.Div(id='binning-metrics-row'),
            
            html.Div([
                ChartContainer.create('binning-price-color-chart', '🎨 Price Colored by Bins', 600),
                ChartContainer.create('binning-analysis-chart', '📊 Bin Analysis Dashboard', 700),
                ChartContainer.create('binning-clustering-chart', '🔍 Clustering Analysis', 500),
                ChartContainer.create('binning-momentum-chart', '📈 Momentum Patterns', 500),
                ChartContainer.create('binning-regime-chart', '🏛️ Market Regime Analysis', 500),
            ])
        ])
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup binning page callbacks"""
        
        ComponentCallbacks.setup_time_range_callback(app, 'binning-time-range')
        
        @app.callback(
            [Output('binning-metrics-row', 'children'),
             Output('binning-price-color-chart', 'figure'),
             Output('binning-analysis-chart', 'figure'),
             Output('binning-clustering-chart', 'figure'),
             Output('binning-momentum-chart', 'figure'),
             Output('binning-regime-chart', 'figure')],
            [Input('binning-time-range-refresh-btn', 'n_clicks')],
            [State('binning-time-range-dropdown', 'value'),
             State('binning-time-range-start-date', 'date'),
             State('binning-time-range-end-date', 'date')]
        )
        def update_binning_page(n_clicks, time_range, start_date, end_date):
            start_time, end_time = ComponentCallbacks.parse_time_range(time_range, start_date, end_date)
            
            try:
                df = self.get_page_data(start_time, end_time)
                
                if df.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
                    empty_fig.update_layout(template='plotly_dark')
                    
                    return (
                        StatusIndicator.create('warning', 'No binning data available for selected time range'),
                        empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
                    )
                
                # Create metrics row with binning-specific metrics
                metrics_row = self.create_metrics_row(df)
                
                binning_metrics = []
                if 'bin_index' in df.columns:
                    unique_bins = df['bin_index'].nunique()
                    most_common_bin = df['bin_index'].mode()[0] if not df['bin_index'].empty else 'N/A'
                    
                    binning_metrics.extend([
                        MetricCard.create(
                            "Active Bins",
                            f"{unique_bins}",
                            f"Most common: Bin {most_common_bin}",
                            '#9B59B6',
                            '📦'
                        ),
                        MetricCard.create(
                            "Bin Distribution",
                            f"{df['bin_index'].std():.2f}",
                            "Standard deviation of bin assignments",
                            '#E67E22',
                            '📊'
                        )
                    ])
                
                if binning_metrics:
                    metrics_row = html.Div([
                        metrics_row,
                        html.Div(binning_metrics, style={
                            'display': 'flex',
                            'gap': '20px',
                            'marginTop': '20px',
                            'flexWrap': 'wrap'
                        })
                    ])
                
                # Create charts
                charts = []
                chart_types = ['price_color', 'bin_analysis', 'clustering_analysis', 'momentum_patterns', 'market_regime']
                
                for chart_type in chart_types:
                    try:
                        fig = self.engine.create_chart(chart_type, 'binning', df)
                        charts.append(fig)
                    except Exception as e:
                        self.logger.error(f"Error creating {chart_type} chart: {e}")
                        empty_fig = go.Figure()
                        empty_fig.add_annotation(text=f"Error loading {chart_type}", xref="paper", yref="paper", x=0.5, y=0.5)
                        empty_fig.update_layout(template='plotly_dark')
                        charts.append(empty_fig)
                
                return [metrics_row] + charts
                
            except Exception as e:
                self.logger.error(f"Error updating binning page: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
                error_fig.update_layout(template='plotly_dark')
                
                return (
                    StatusIndicator.create('error', f'Error loading binning data: {str(e)}'),
                    error_fig, error_fig, error_fig, error_fig, error_fig
                )


class DominanceFeaturePage(BasePage):
    """Dominance Features Page"""
    
    category = 'dominance'
    
    def create_layout(self) -> html.Div:
        """Create dominance features page layout"""
        
        return html.Div([
            PageHeader.create(
                "👑 Market Dominance Features",
                "Analysis of market dominance patterns, regime transitions, and liquidation-based market control indicators",
                7  # Number of dominance features
            ),
            
            TimeRangeSelector.create('dominance-time-range'),
            
            html.Div(id='dominance-metrics-row'),
            
            html.Div([
                ChartContainer.create('dominance-status-chart', '📊 Dominance Status Overview', 700),
                ChartContainer.create('dominance-regime-chart', '🏛️ Market Regime Dashboard', 600),
                ChartContainer.create('dominance-patterns-chart', '📈 Dominance Patterns Analysis', 500),
                ChartContainer.create('dominance-correlation-chart', '🔗 Dominance Correlation Matrix', 500),
            ])
        ])
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup dominance page callbacks"""
        
        ComponentCallbacks.setup_time_range_callback(app, 'dominance-time-range')
        
        @app.callback(
            [Output('dominance-metrics-row', 'children'),
             Output('dominance-status-chart', 'figure'),
             Output('dominance-regime-chart', 'figure'),
             Output('dominance-patterns-chart', 'figure'),
             Output('dominance-correlation-chart', 'figure')],
            [Input('dominance-time-range-refresh-btn', 'n_clicks')],
            [State('dominance-time-range-dropdown', 'value'),
             State('dominance-time-range-start-date', 'date'),
             State('dominance-time-range-end-date', 'date')]
        )
        def update_dominance_page(n_clicks, time_range, start_date, end_date):
            start_time, end_time = ComponentCallbacks.parse_time_range(time_range, start_date, end_date)
            
            try:
                df = self.get_page_data(start_time, end_time)
                
                if df.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
                    empty_fig.update_layout(template='plotly_dark')
                    
                    return (
                        StatusIndicator.create('warning', 'No dominance data available for selected time range'),
                        empty_fig, empty_fig, empty_fig, empty_fig
                    )
                
                # Create metrics row with dominance-specific metrics
                metrics_row = self.create_metrics_row(df)
                
                dominance_metrics = []
                if 'dominance' in df.columns:
                    bull_dominance = (df['dominance'] == 1).sum()
                    bear_dominance = (df['dominance'] == -1).sum()
                    neutral = (df['dominance'] == 0).sum()
                    
                    dominance_metrics.extend([
                        MetricCard.create(
                            "Bull Dominance",
                            f"{bull_dominance:,}",
                            f"{bull_dominance/len(df)*100:.1f}% of time",
                            '#10B981',
                            '🐂'
                        ),
                        MetricCard.create(
                            "Bear Dominance", 
                            f"{bear_dominance:,}",
                            f"{bear_dominance/len(df)*100:.1f}% of time",
                            '#EF4444',
                            '🐻'
                        ),
                        MetricCard.create(
                            "Neutral Periods",
                            f"{neutral:,}",
                            f"{neutral/len(df)*100:.1f}% of time",
                            '#6B7280',
                            '⚖️'
                        )
                    ])
                
                if 'dominance_duration_total' in df.columns:
                    avg_duration = df['dominance_duration_total'].mean()
                    dominance_metrics.append(
                        MetricCard.create(
                            "Avg Duration",
                            f"{avg_duration:.1f}h",
                            f"Max: {df['dominance_duration_total'].max():.1f}h",
                            '#3B82F6',
                            '⏱️'
                        )
                    )
                
                if dominance_metrics:
                    metrics_row = html.Div([
                        metrics_row,
                        html.Div(dominance_metrics, style={
                            'display': 'flex',
                            'gap': '20px',
                            'marginTop': '20px',
                            'flexWrap': 'wrap'
                        })
                    ])
                
                # Create charts
                charts = []
                chart_types = ['dominance_status', 'market_regime', 'dominance_patterns', 'dominance_correlation']
                
                for chart_type in chart_types:
                    try:
                        fig = self.engine.create_chart(chart_type, 'dominance', df)
                        charts.append(fig)
                    except Exception as e:
                        self.logger.error(f"Error creating {chart_type} chart: {e}")
                        empty_fig = go.Figure()
                        empty_fig.add_annotation(text=f"Error loading {chart_type}", xref="paper", yref="paper", x=0.5, y=0.5)
                        empty_fig.update_layout(template='plotly_dark')
                        charts.append(empty_fig)
                
                return [metrics_row] + charts
                
            except Exception as e:
                self.logger.error(f"Error updating dominance page: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
                error_fig.update_layout(template='plotly_dark')
                
                return (
                    StatusIndicator.create('error', f'Error loading dominance data: {str(e)}'),
                    error_fig, error_fig, error_fig, error_fig
                )


class SignalFeaturePage(BasePage):
    """Signal Features Page"""
    
    category = 'signals'
    
    def create_layout(self) -> html.Div:
        """Create signal features page layout"""
        
        return html.Div([
            PageHeader.create(
                "📡 Trading Signal Features",
                "Comprehensive trading signal analysis with ceiling/bottom detection, reversal patterns, and performance metrics",
                5  # Number of signal features
            ),
            
            TimeRangeSelector.create('signals-time-range'),
            
            html.Div(id='signals-metrics-row'),
            
            html.Div([
                ChartContainer.create('signals-main-chart', '📡 Trading Signals Overview', 600),
                ChartContainer.create('signals-liquidations-chart', '💥 Liquidations Analysis', 500),
                ChartContainer.create('signals-trading-chart', '📊 Trading Signals Dashboard', 700),
                ChartContainer.create('signals-ceiling-chart', '🏔️ Ceiling/Bottom Analysis', 500),
                ChartContainer.create('signals-reversal-chart', '🔄 Reversal Signals Analysis', 500),
                ChartContainer.create('signals-performance-chart', '📈 Signal Performance Metrics', 600),
            ])
        ])
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup signals page callbacks"""
        
        ComponentCallbacks.setup_time_range_callback(app, 'signals-time-range')
        
        @app.callback(
            [Output('signals-metrics-row', 'children'),
             Output('signals-main-chart', 'figure'),
             Output('signals-liquidations-chart', 'figure'),
             Output('signals-trading-chart', 'figure'),
             Output('signals-ceiling-chart', 'figure'),
             Output('signals-reversal-chart', 'figure'),
             Output('signals-performance-chart', 'figure')],
            [Input('signals-time-range-refresh-btn', 'n_clicks')],
            [State('signals-time-range-dropdown', 'value'),
             State('signals-time-range-start-date', 'date'),
             State('signals-time-range-end-date', 'date')]
        )
        def update_signals_page(n_clicks, time_range, start_date, end_date):
            start_time, end_time = ComponentCallbacks.parse_time_range(time_range, start_date, end_date)
            
            try:
                df = self.get_page_data(start_time, end_time)
                
                if df.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
                    empty_fig.update_layout(template='plotly_dark')
                    
                    return (
                        StatusIndicator.create('warning', 'No signals data available for selected time range'),
                        empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
                    )
                
                # Create metrics row with signals-specific metrics
                metrics_row = self.create_metrics_row(df)
                
                signals_metrics = []
                
                # Ceiling/Bottom signals
                if 'hit_ceiling_bottom' in df.columns:
                    ceiling_hits = (df['hit_ceiling_bottom'] == -1).sum()
                    bottom_hits = (df['hit_ceiling_bottom'] == 1).sum()
                    
                    signals_metrics.extend([
                        MetricCard.create(
                            "Ceiling Hits",
                            f"{ceiling_hits:,}",
                            f"{ceiling_hits/len(df)*100:.2f}% of data points",
                            '#EF4444',
                            '🔺'
                        ),
                        MetricCard.create(
                            "Bottom Hits",
                            f"{bottom_hits:,}",
                            f"{bottom_hits/len(df)*100:.2f}% of data points",
                            '#10B981',
                            '🔻'
                        )
                    ])
                
                # Signal class distribution
                if 'signal_class' in df.columns:
                    active_signals = (df['signal_class'] != 'Neutral').sum()
                    buy_signals = df['signal_class'].str.contains('Buy', na=False).sum()
                    sell_signals = df['signal_class'].str.contains('Sell', na=False).sum()
                    
                    signals_metrics.extend([
                        MetricCard.create(
                            "Active Signals",
                            f"{active_signals:,}",
                            f"{active_signals/len(df)*100:.1f}% activity rate",
                            '#3B82F6',
                            '⚡'
                        ),
                        MetricCard.create(
                            "Buy/Sell Ratio",
                            f"{buy_signals}/{sell_signals}",
                            f"Buy: {buy_signals/max(active_signals,1)*100:.1f}%",
                            '#F59E0B',
                            '⚖️'
                        )
                    ])
                
                if signals_metrics:
                    metrics_row = html.Div([
                        metrics_row,
                        html.Div(signals_metrics, style={
                            'display': 'flex',
                            'gap': '20px',
                            'marginTop': '20px',
                            'flexWrap': 'wrap'
                        })
                    ])
                
                # Create charts
                charts = []
                chart_types = ['signals_plot', 'liquidations_analysis', 'trading_signals', 'ceiling_bottom', 'reversal_signals', 'signal_performance']
                
                for chart_type in chart_types:
                    try:
                        fig = self.engine.create_chart(chart_type, 'signals', df)
                        charts.append(fig)
                    except Exception as e:
                        self.logger.error(f"Error creating {chart_type} chart: {e}")
                        empty_fig = go.Figure()
                        empty_fig.add_annotation(text=f"Error loading {chart_type}", xref="paper", yref="paper", x=0.5, y=0.5)
                        empty_fig.update_layout(template='plotly_dark')
                        charts.append(empty_fig)
                
                return [metrics_row] + charts
                
            except Exception as e:
                self.logger.error(f"Error updating signals page: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
                error_fig.update_layout(template='plotly_dark')
                
                return (
                    StatusIndicator.create('error', f'Error loading signals data: {str(e)}'),
                    error_fig, error_fig, error_fig, error_fig, error_fig, error_fig
                )


class AdvancedFeaturePage(BasePage):
    """Advanced Features Page"""
    
    category = 'advanced'
    
    def create_layout(self) -> html.Div:
        """Create advanced features page layout"""
        
        return html.Div([
            PageHeader.create(
                "🚀 Advanced Features",
                "Technical indicators, volatility models, and advanced statistical features for sophisticated market analysis",
                10  # Number of advanced features
            ),
            
            TimeRangeSelector.create('advanced-time-range'),
            
            html.Div(id='advanced-metrics-row'),
            
            html.Div([
                ChartContainer.create('advanced-main-chart', '🚀 Advanced Features Dashboard', 700),
                ChartContainer.create('advanced-volatility-chart', '📊 Volatility Analysis', 500),
                ChartContainer.create('advanced-momentum-chart', '📈 Momentum Indicators', 500),
                ChartContainer.create('advanced-technical-chart', '🔧 Technical Indicators', 600),
            ])
        ])
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup advanced features page callbacks"""
        
        ComponentCallbacks.setup_time_range_callback(app, 'advanced-time-range')
        
        @app.callback(
            [Output('advanced-metrics-row', 'children'),
             Output('advanced-main-chart', 'figure'),
             Output('advanced-volatility-chart', 'figure'),
             Output('advanced-momentum-chart', 'figure'),
             Output('advanced-technical-chart', 'figure')],
            [Input('advanced-time-range-refresh-btn', 'n_clicks')],
            [State('advanced-time-range-dropdown', 'value'),
             State('advanced-time-range-start-date', 'date'),
             State('advanced-time-range-end-date', 'date')]
        )
        def update_advanced_page(n_clicks, time_range, start_date, end_date):
            start_time, end_time = ComponentCallbacks.parse_time_range(time_range, start_date, end_date)
            
            try:
                df = self.get_page_data(start_time, end_time)
                
                if df.empty:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="No advanced features data available yet", xref="paper", yref="paper", x=0.5, y=0.5)
                    empty_fig.update_layout(template='plotly_dark')
                    
                    return (
                        StatusIndicator.create('info', 'Advanced features data not available yet. Please run the advanced feature engineering pipeline.'),
                        empty_fig, empty_fig, empty_fig, empty_fig
                    )
                
                # Create metrics row
                metrics_row = self.create_metrics_row(df)
                
                # Create basic charts (advanced features may not have specific visualizers yet)
                charts = []
                
                # Main dashboard - comparison of key advanced features
                comparison_fig = self.engine.create_comparison_dashboard(['advanced'], start_time, end_time)
                charts.append(comparison_fig)
                
                # Create basic time series plots for other charts
                for title in ['Volatility Analysis', 'Momentum Indicators', 'Technical Indicators']:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['time'],
                        y=df['price'],
                        mode='lines',
                        name='BTC Price',
                        line=dict(color='#00D4FF')
                    ))
                    fig.update_layout(
                        title=title,
                        template='plotly_dark',
                        height=400
                    )
                    charts.append(fig)
                
                return [metrics_row] + charts
                
            except Exception as e:
                self.logger.error(f"Error updating advanced page: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
                error_fig.update_layout(template='plotly_dark')
                
                return (
                    StatusIndicator.create('error', f'Error loading advanced features: {str(e)}'),
                    error_fig, error_fig, error_fig, error_fig
                )


class StatisticalAnalysisPage(BasePage):
    """Statistical Analysis Page"""
    
    category = 'statistical'
    
    def create_layout(self) -> html.Div:
        """Create statistical analysis page layout"""
        
        return html.Div([
            PageHeader.create(
                "📈 Statistical Analysis",
                "Comprehensive statistical analysis with distribution fitting, normality tests, and risk metrics for all feature categories"
            ),
            
            TimeRangeSelector.create('stats-time-range'),
            
            html.Div(id='stats-controls', children=[
                html.Div([
                    html.Label("📊 Analysis Categories:", style={'color': 'white', 'fontWeight': '500'}),
                    dcc.Checklist(
                        id='stats-categories-checklist',
                        options=[
                            {'label': ' 🎯 RPN Features', 'value': 'rpn'},
                            {'label': ' 📦 Binning Features', 'value': 'bin'},
                            {'label': ' 👑 Dominance Features', 'value': 'dominance'},
                            {'label': ' 📡 Signal Features', 'value': 'signals'},
                            {'label': ' 🚀 Advanced Features', 'value': 'advanced'}
                        ],
                        value=['rpn', 'bin', 'dominance', 'signals'],
                        style={'color': 'white'},
                        inputStyle={'marginRight': '10px'},
                        labelStyle={'display': 'block', 'marginBottom': '8px'}
                    )
                ], style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'marginBottom': '20px',
                    'border': '1px solid rgba(0, 212, 255, 0.2)'
                })
            ]),
            
            html.Div(id='stats-summary-row'),
            
            html.Div([
                ChartContainer.create('stats-overview-chart', '📊 Statistical Overview Dashboard', 700),
                ChartContainer.create('stats-distribution-chart', '📈 Distribution Analysis', 600),
                ChartContainer.create('stats-correlation-chart', '🔗 Feature Correlations', 600),
                ChartContainer.create('stats-risk-chart', '⚠️ Risk Metrics Analysis', 500),
            ])
        ])
    
    def setup_callbacks(self, app: dash.Dash):
        """Setup statistical analysis page callbacks"""
        
        ComponentCallbacks.setup_time_range_callback(app, 'stats-time-range')
        
        @app.callback(
            [Output('stats-summary-row', 'children'),
             Output('stats-overview-chart', 'figure'),
             Output('stats-distribution-chart', 'figure'),
             Output('stats-correlation-chart', 'figure'),
             Output('stats-risk-chart', 'figure')],
            [Input('stats-time-range-refresh-btn', 'n_clicks')],
            [State('stats-time-range-dropdown', 'value'),
             State('stats-time-range-start-date', 'date'),
             State('stats-time-range-end-date', 'date'),
             State('stats-categories-checklist', 'value')]
        )
        def update_stats_page(n_clicks, time_range, start_date, end_date, selected_categories):
            start_time, end_time = ComponentCallbacks.parse_time_range(time_range, start_date, end_date)
            
            try:
                # Get statistical analysis
                analysis_results = self.engine.get_statistical_analysis(start_time, end_time, selected_categories)
                
                if 'error' in analysis_results:
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(text="Error in statistical analysis", xref="paper", yref="paper", x=0.5, y=0.5)
                    empty_fig.update_layout(template='plotly_dark')
                    
                    return (
                        StatusIndicator.create('error', f'Statistical analysis failed: {analysis_results["error"]}'),
                        empty_fig, empty_fig, empty_fig, empty_fig
                    )
                
                # Create summary metrics
                summary = analysis_results.get('executive_summary', {})
                
                summary_metrics = [
                    MetricCard.create(
                        "Categories Analyzed",
                        f"{summary.get('total_categories', 0)}",
                        f"From {len(selected_categories)} selected",
                        '#3B82F6',
                        '📊'
                    ),
                    MetricCard.create(
                        "Features Analyzed",
                        f"{summary.get('successful_analyses', 0)}",
                        f"Success rate: {summary.get('successful_analyses', 0) / max(summary.get('total_features', 1), 1) * 100:.1f}%",
                        '#10B981',
                        '✅'
                    ),
                    MetricCard.create(
                        "Analysis Time",
                        f"{datetime.now().strftime('%H:%M')}",
                        f"Range: {start_time.strftime('%m/%d')} - {end_time.strftime('%m/%d')}",
                        '#F59E0B',
                        '⏰'
                    )
                ]
                
                summary_row = html.Div(summary_metrics, style={
                    'display': 'flex',
                    'gap': '20px',
                    'marginBottom': '30px',
                    'flexWrap': 'wrap'
                })
                
                # Create statistical charts (placeholder implementations)
                charts = []
                
                # Overview chart
                overview_fig = go.Figure()
                if summary.get('key_insights'):
                    # Create insights visualization
                    insights = summary['key_insights']
                    for i, insight in enumerate(insights[:3]):  # Show top 3 insights
                        overview_fig.add_trace(go.Bar(
                            x=insight['features'][:5],  # Top 5 features
                            y=insight['values'][:5],
                            name=insight['type'].replace('_', ' ').title(),
                            yaxis=f'y{i+1}' if i > 0 else 'y'
                        ))
                
                overview_fig.update_layout(
                    title='Statistical Analysis Overview',
                    template='plotly_dark',
                    height=500
                )
                charts.append(overview_fig)
                
                # Placeholder charts for other visualizations
                for title in ['Distribution Analysis', 'Feature Correlations', 'Risk Metrics Analysis']:
                    fig = go.Figure()
                    fig.add_annotation(
                        text=f"Statistical {title} visualization",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="white")
                    )
                    fig.update_layout(
                        title=title,
                        template='plotly_dark',
                        height=400
                    )
                    charts.append(fig)
                
                return [summary_row] + charts
                
            except Exception as e:
                self.logger.error(f"Error updating statistical analysis page: {e}")
                error_fig = go.Figure()
                error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5)
                error_fig.update_layout(template='plotly_dark')
                
                return (
                    StatusIndicator.create('error', f'Error in statistical analysis: {str(e)}'),
                    error_fig, error_fig, error_fig, error_fig
                )