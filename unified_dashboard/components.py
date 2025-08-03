"""
Reusable Dashboard Components

Common UI components used across all feature pages.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable


class NavigationBar:
    """Professional navigation bar for the unified dashboard"""
    
    @staticmethod
    def create(current_page: str = 'overview') -> html.Div:
        """
        Create navigation bar component
        
        Args:
            current_page: Currently active page
            
        Returns:
            Navigation bar HTML component
        """
        
        nav_items = [
            {'label': '📊 Overview', 'value': 'overview', 'href': '/'},
            {'label': '🎯 RPN Features', 'value': 'rpn', 'href': '/rpn'},
            {'label': '📦 Binning Features', 'value': 'binning', 'href': '/binning'},
            {'label': '👑 Dominance Features', 'value': 'dominance', 'href': '/dominance'},
            {'label': '📡 Signal Features', 'value': 'signals', 'href': '/signals'},
            {'label': '🚀 Advanced Features', 'value': 'advanced', 'href': '/advanced'},
            {'label': '📈 Statistical Analysis', 'value': 'statistical', 'href': '/statistical'},
            {'label': '🔗 API', 'value': 'api', 'href': '/api'}
        ]
        
        nav_links = []
        for item in nav_items:
            is_active = item['value'] == current_page
            
            link_style = {
                'color': '#00D4FF' if is_active else '#9CA3AF',
                'textDecoration': 'none',
                'padding': '10px 20px',
                'borderRadius': '8px',
                'backgroundColor': 'rgba(0, 212, 255, 0.1)' if is_active else 'transparent',
                'border': '1px solid rgba(0, 212, 255, 0.3)' if is_active else '1px solid transparent',
                'transition': 'all 0.3s ease',
                'display': 'inline-block',
                'marginRight': '10px',
                'marginBottom': '10px',
                'fontSize': '14px',
                'fontWeight': '500'
            }
            
            nav_links.append(
                dcc.Link(
                    item['label'],
                    href=item['href'],
                    style=link_style,
                    className=f'nav-link {"active" if is_active else ""}'
                )
            )
        
        return html.Div([
            html.Div([
                html.H2("🔬 BTC Advanced Feature Dashboard", 
                       style={
                           'margin': '0 0 20px 0',
                           'color': 'white',
                           'fontSize': '24px',
                           'fontWeight': '600'
                       }),
                html.P("Comprehensive analysis and visualization of Bitcoin trading features",
                      style={
                          'margin': '0 0 30px 0',
                          'color': '#9CA3AF',
                          'fontSize': '16px'
                      })
            ]),
            
            html.Div(nav_links, style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'alignItems': 'center'
            })
            
        ], style={
            'background': 'linear-gradient(135deg, rgba(0, 0, 0, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%)',
            'backdropFilter': 'blur(10px)',
            'borderRadius': '15px',
            'padding': '30px',
            'marginBottom': '30px',
            'border': '1px solid rgba(0, 212, 255, 0.2)',
            'boxShadow': '0 8px 32px rgba(0, 212, 255, 0.1)'
        })


class TimeRangeSelector:
    """Time range selection component"""
    
    @staticmethod
    def create(component_id: str = 'time-range-selector') -> html.Div:
        """
        Create time range selector component
        
        Args:
            component_id: Unique ID for the component
            
        Returns:
            Time range selector HTML component
        """
        
        return html.Div([
            html.Label("📅 Time Range:", 
                      style={'color': 'white', 'marginRight': '15px', 'fontWeight': '500'}),
            
            dcc.Dropdown(
                id=f'{component_id}-dropdown',
                options=[
                    {'label': '🕐 Last Hour', 'value': '1h'},
                    {'label': '🕕 Last 6 Hours', 'value': '6h'},
                    {'label': '📅 Last 24 Hours', 'value': '24h'},
                    {'label': '📅 Last 3 Days', 'value': '3d'},
                    {'label': '📅 Last Week', 'value': '7d'},
                    {'label': '📅 Last Month', 'value': '30d'},
                    {'label': '📅 Last 3 Months', 'value': '90d'},
                    {'label': '⚙️ Custom Range', 'value': 'custom'}
                ],
                value='24h',
                style={
                    'width': '200px',
                    'backgroundColor': 'rgba(255, 255, 255, 0.1)',
                    'color': 'white'
                },
                className='time-range-dropdown'
            ),
            
            html.Div([
                html.Label("From:", style={'color': 'white', 'marginRight': '10px'}),
                dcc.DatePickerSingle(
                    id=f'{component_id}-start-date',
                    date=(datetime.now() - timedelta(days=7)).date(),
                    display_format='YYYY-MM-DD',
                    style={'marginRight': '20px'}
                ),
                
                html.Label("To:", style={'color': 'white', 'marginRight': '10px'}),
                dcc.DatePickerSingle(
                    id=f'{component_id}-end-date',
                    date=datetime.now().date(),
                    display_format='YYYY-MM-DD'
                )
            ], 
            id=f'{component_id}-custom-container',
            style={'display': 'none', 'marginTop': '15px', 'alignItems': 'center'},
            className='custom-date-container'
            ),
            
            html.Button(
                '🔄 Refresh Data',
                id=f'{component_id}-refresh-btn',
                style={
                    'marginLeft': '20px',
                    'padding': '8px 16px',
                    'backgroundColor': '#00D4FF',
                    'color': 'black',
                    'border': 'none',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontWeight': '500',
                    'fontSize': '14px'
                },
                className='refresh-button'
            )
            
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'flexWrap': 'wrap',
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '30px',
            'border': '1px solid rgba(0, 212, 255, 0.2)'
        })


class FeatureSelector:
    """Feature category selection component"""
    
    @staticmethod
    def create(component_id: str = 'feature-selector',
               available_features: Optional[List[str]] = None) -> html.Div:
        """
        Create feature selector component
        
        Args:
            component_id: Unique ID for the component
            available_features: List of available feature categories
            
        Returns:
            Feature selector HTML component
        """
        
        if available_features is None:
            available_features = ['rpn', 'binning', 'dominance', 'signals', 'advanced']
        
        feature_options = []
        feature_icons = {
            'rpn': '🎯',
            'binning': '📦', 
            'dominance': '👑',
            'signals': '📡',
            'advanced': '🚀',
            'statistical': '📈'
        }
        
        for feature in available_features:
            icon = feature_icons.get(feature, '📊')
            label = f"{icon} {feature.replace('_', ' ').title()} Features"
            feature_options.append({'label': label, 'value': feature})
        
        return html.Div([
            html.Label("🔧 Feature Categories:", 
                      style={'color': 'white', 'marginBottom': '10px', 'fontWeight': '500'}),
            
            dcc.Checklist(
                id=f'{component_id}-checklist',
                options=feature_options,
                value=available_features,
                style={'color': 'white'},
                inputStyle={'marginRight': '10px'},
                labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
            )
            
        ], style={
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '20px',
            'border': '1px solid rgba(0, 212, 255, 0.2)'
        })


class MetricCard:
    """Reusable metric display card"""
    
    @staticmethod
    def create(title: str, value: str, subtitle: str = "", 
               color: str = '#00D4FF', icon: str = "📊") -> html.Div:
        """
        Create metric card component
        
        Args:
            title: Card title
            value: Main metric value
            subtitle: Additional information
            color: Accent color
            icon: Icon emoji
            
        Returns:
            Metric card HTML component
        """
        
        return html.Div([
            html.Div([
                html.Span(icon, style={'fontSize': '24px', 'marginRight': '10px'}),
                html.H4(title, style={
                    'margin': '0',
                    'color': '#9CA3AF',
                    'fontSize': '14px',
                    'fontWeight': '500'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
            
            html.H2(value, style={
                'margin': '0 0 5px 0',
                'color': color,
                'fontSize': '28px',
                'fontWeight': '700'
            }),
            
            html.P(subtitle, style={
                'margin': '0',
                'color': '#6B7280',
                'fontSize': '12px'
            }) if subtitle else None
            
        ], style={
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'padding': '20px',
            'borderRadius': '10px',
            'border': f'1px solid {color}30',  # 30 for alpha
            'backdropFilter': 'blur(5px)',
            'minWidth': '200px',
            'textAlign': 'left'
        })


class ChartContainer:
    """Container for charts with loading and error states"""
    
    @staticmethod
    def create(chart_id: str, title: str, height: int = 500, 
               loading_text: str = "Loading chart...") -> html.Div:
        """
        Create chart container component
        
        Args:
            chart_id: Unique ID for the chart
            title: Chart title
            height: Chart height in pixels
            loading_text: Text to show while loading
            
        Returns:
            Chart container HTML component
        """
        
        return html.Div([
            html.H3(title, style={
                'color': 'white',
                'marginBottom': '20px',
                'fontSize': '20px',
                'fontWeight': '600'
            }),
            
            dcc.Loading(
                id=f'{chart_id}-loading',
                type='circle',
                color='#00D4FF',
                children=[
                    dcc.Graph(
                        id=chart_id,
                        style={'height': f'{height}px'},
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                            'responsive': True
                        }
                    )
                ]
            )
            
        ], style={
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'padding': '25px',
            'borderRadius': '12px',
            'marginBottom': '30px',
            'border': '1px solid rgba(0, 212, 255, 0.2)',
            'backdropFilter': 'blur(5px)'
        })


class StatusIndicator:
    """Status indicator component"""
    
    @staticmethod
    def create(status: str, message: str = "") -> html.Div:
        """
        Create status indicator component
        
        Args:
            status: Status type ('success', 'warning', 'error', 'info')
            message: Status message
            
        Returns:
            Status indicator HTML component
        """
        
        status_config = {
            'success': {'color': '#10B981', 'icon': '✅', 'bg': 'rgba(16, 185, 129, 0.1)'},
            'warning': {'color': '#F59E0B', 'icon': '⚠️', 'bg': 'rgba(245, 158, 11, 0.1)'},
            'error': {'color': '#EF4444', 'icon': '❌', 'bg': 'rgba(239, 68, 68, 0.1)'},
            'info': {'color': '#3B82F6', 'icon': 'ℹ️', 'bg': 'rgba(59, 130, 246, 0.1)'},
            'loading': {'color': '#00D4FF', 'icon': '⏳', 'bg': 'rgba(0, 212, 255, 0.1)'}
        }
        
        config = status_config.get(status, status_config['info'])
        
        return html.Div([
            html.Span(config['icon'], style={'marginRight': '10px', 'fontSize': '16px'}),
            html.Span(message, style={'color': config['color'], 'fontWeight': '500'})
        ], style={
            'backgroundColor': config['bg'],
            'border': f'1px solid {config["color"]}40',
            'borderRadius': '8px',
            'padding': '12px 16px',
            'marginBottom': '20px',
            'display': 'flex',
            'alignItems': 'center'
        })


class PageHeader:
    """Standard page header component"""
    
    @staticmethod
    def create(title: str, description: str = "", 
               feature_count: Optional[int] = None) -> html.Div:
        """
        Create page header component
        
        Args:
            title: Page title
            description: Page description
            feature_count: Number of features (optional)
            
        Returns:
            Page header HTML component
        """
        
        return html.Div([
            html.H1(title, style={
                'color': 'white',
                'margin': '0 0 10px 0',
                'fontSize': '32px',
                'fontWeight': '700',
                'background': 'linear-gradient(90deg, #00D4FF, #4ECDC4)',
                'backgroundClip': 'text',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent'
            }),
            
            html.P(description, style={
                'color': '#9CA3AF',
                'margin': '0 0 15px 0',
                'fontSize': '16px',
                'lineHeight': '1.6'
            }) if description else None,
            
            html.Div([
                html.Span(f"📊 {feature_count} features available", style={
                    'color': '#00D4FF',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'backgroundColor': 'rgba(0, 212, 255, 0.1)',
                    'padding': '4px 8px',
                    'borderRadius': '4px',
                    'border': '1px solid rgba(0, 212, 255, 0.3)'
                })
            ]) if feature_count is not None else None
            
        ], style={
            'marginBottom': '30px',
            'paddingBottom': '20px',
            'borderBottom': '1px solid rgba(0, 212, 255, 0.2)'
        })


# Callback utilities for component interactions
class ComponentCallbacks:
    """Utilities for setting up component callbacks"""
    
    @staticmethod
    def setup_time_range_callback(app: dash.Dash, component_id: str = 'time-range-selector'):
        """
        Setup callback for time range selector
        
        Args:
            app: Dash application instance
            component_id: Component ID prefix
        """
        
        @app.callback(
            Output(f'{component_id}-custom-container', 'style'),
            Input(f'{component_id}-dropdown', 'value')
        )
        def toggle_custom_date_picker(time_range):
            if time_range == 'custom':
                return {'display': 'flex', 'marginTop': '15px', 'alignItems': 'center'}
            return {'display': 'none'}
    
    @staticmethod
    def parse_time_range(time_range: str, start_date: str = None, 
                        end_date: str = None) -> tuple:
        """
        Parse time range selection into start/end datetime objects
        
        Args:
            time_range: Time range selection ('1h', '24h', 'custom', etc.)
            start_date: Custom start date (for 'custom' range)
            end_date: Custom end date (for 'custom' range)
            
        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        
        end_time = datetime.now()
        
        if time_range == 'custom' and start_date and end_date:
            start_time = datetime.strptime(start_date, '%Y-%m-%d')
            end_time = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            time_mappings = {
                '1h': 1,
                '6h': 6, 
                '24h': 24,
                '3d': 72,
                '7d': 168,
                '30d': 720,
                '90d': 2160
            }
            
            hours = time_mappings.get(time_range, 24)
            start_time = end_time - timedelta(hours=hours)
        
        return start_time, end_time