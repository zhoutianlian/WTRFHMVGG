"""
Unified Main Dashboard Application

The main application that orchestrates all feature pages and provides
a cohesive website experience with separate pages for each feature category.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'feature_engineer'))

from .components import NavigationBar
from .visualization_engine import VisualizationEngine
from .feature_pages import (
    RPNFeaturePage, BinningFeaturePage, DominanceFeaturePage,
    SignalFeaturePage, AdvancedFeaturePage, StatisticalAnalysisPage
)


class UnifiedDashboard:
    """
    Main unified dashboard application
    
    Combines all visualization functions from different modules into
    a single cohesive website with separate pages for each feature category.
    
    Features:
    - 🎯 RPN Features: Risk Priority Number analysis
    - 📦 Binning Features: K-means clustering and momentum
    - 👑 Dominance Features: Market dominance and regime classification  
    - 📡 Signal Features: Trading signals and ceiling/bottom detection
    - 🚀 Advanced Features: Technical indicators and volatility models
    - 📈 Statistical Analysis: Comprehensive statistical analysis
    """
    
    def __init__(self, db_config: Optional[Dict] = None, 
                 port: int = 8050, debug: bool = False):
        """
        Initialize the unified dashboard
        
        Args:
            db_config: Database configuration
            port: Port for the web server
            debug: Enable debug mode
        """
        self.port = port
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization engine
        self.engine = VisualizationEngine(db_config)
        
        # Initialize Dash app with professional styling
        self.app = dash.Dash(
            __name__,
            title="BTC Advanced Feature Dashboard",
            suppress_callback_exceptions=True,
            external_stylesheets=[
                'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
            ]
        )
        
        # Initialize feature pages
        self.pages = {
            'rpn': RPNFeaturePage(self.engine),
            'binning': BinningFeaturePage(self.engine),
            'dominance': DominanceFeaturePage(self.engine),
            'signals': SignalFeaturePage(self.engine),
            'advanced': AdvancedFeaturePage(self.engine),
            'statistical': StatisticalAnalysisPage(self.engine)
        }
        
        # Setup the application
        self._setup_layout()
        self._setup_callbacks()
        
        self.logger.info("Unified dashboard initialized successfully")
    
    def _setup_layout(self):
        """Setup the main application layout"""
        
        # Custom CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: 'Inter', sans-serif;
                        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                        color: #ffffff;
                        min-height: 100vh;
                        overflow-x: hidden;
                    }
                    
                    .main-container {
                        background: rgba(0, 0, 0, 0.8);
                        backdrop-filter: blur(20px);
                        min-height: 100vh;
                        padding: 20px;
                    }
                    
                    .content-wrapper {
                        max-width: 1400px;
                        margin: 0 auto;
                    }
                    
                    /* Navigation styles */
                    .nav-link:hover {
                        background-color: rgba(0, 212, 255, 0.2) !important;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
                    }
                    
                    /* Custom scrollbar */
                    ::-webkit-scrollbar {
                        width: 8px;
                    }
                    
                    ::-webkit-scrollbar-track {
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 4px;
                    }
                    
                    ::-webkit-scrollbar-thumb {
                        background: linear-gradient(45deg, #00D4FF, #4ECDC4);
                        border-radius: 4px;
                    }
                    
                    ::-webkit-scrollbar-thumb:hover {
                        background: linear-gradient(45deg, #4ECDC4, #00D4FF);
                    }
                    
                    /* Loading animations */
                    .loading-spinner {
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    
                    /* Chart container enhancements */
                    .chart-container {
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                    }
                    
                    .chart-container:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 12px 32px rgba(0, 212, 255, 0.2);
                    }
                    
                    /* Dropdown styling */
                    .Select-control {
                        background-color: rgba(255, 255, 255, 0.1) !important;
                        border: 1px solid rgba(0, 212, 255, 0.3) !important;
                        color: white !important;
                    }
                    
                    .Select-menu-outer {
                        background-color: rgba(26, 26, 46, 0.95) !important;
                        border: 1px solid rgba(0, 212, 255, 0.3) !important;
                    }
                    
                    .Select-option {
                        background-color: transparent !important;
                        color: white !important;
                    }
                    
                    .Select-option:hover {
                        background-color: rgba(0, 212, 255, 0.2) !important;
                    }
                    
                    /* Button styling */
                    .refresh-button:hover {
                        background-color: #4ECDC4 !important;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
                    }
                    
                    /* Status indicators */
                    .status-success {
                        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2));
                        border-left: 4px solid #10B981;
                    }
                    
                    .status-warning {
                        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.2));
                        border-left: 4px solid #F59E0B;
                    }
                    
                    .status-error {
                        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2));
                        border-left: 4px solid #EF4444;
                    }
                    
                    /* Responsive design */
                    @media (max-width: 768px) {
                        .main-container {
                            padding: 10px;
                        }
                        
                        .content-wrapper {
                            padding: 0 10px;
                        }
                        
                        .nav-link {
                            font-size: 12px !important;
                            padding: 8px 12px !important;
                            margin-right: 5px !important;
                        }
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Main layout
        self.app.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            
            html.Div([
                html.Div([
                    # Navigation will be updated based on current page
                    html.Div(id='navigation-container'),
                    
                    # Page content
                    html.Div(id='page-content')
                    
                ], className='content-wrapper')
                
            ], className='main-container')
        ])
    
    def _setup_callbacks(self):
        """Setup main application callbacks"""
        
        # Main routing callback
        @self.app.callback(
            [Output('navigation-container', 'children'),
             Output('page-content', 'children')],
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            """Route to appropriate page based on URL"""
            
            # Determine current page
            if pathname == '/rpn':
                current_page = 'rpn'
                page_content = self.pages['rpn'].create_layout()
            elif pathname == '/binning':
                current_page = 'binning'
                page_content = self.pages['binning'].create_layout()
            elif pathname == '/dominance':
                current_page = 'dominance'
                page_content = self.pages['dominance'].create_layout()
            elif pathname == '/signals':
                current_page = 'signals'
                page_content = self.pages['signals'].create_layout()
            elif pathname == '/advanced':
                current_page = 'advanced'
                page_content = self.pages['advanced'].create_layout()
            elif pathname == '/statistical':
                current_page = 'statistical'
                page_content = self.pages['statistical'].create_layout()
            elif pathname == '/api':
                current_page = 'api'
                page_content = self._create_api_page()
            else:
                current_page = 'overview'
                page_content = self._create_overview_page()
            
            # Create navigation with current page highlighted
            navigation = NavigationBar.create(current_page)
            
            return navigation, page_content
        
        # Setup callbacks for each page
        for page in self.pages.values():
            page.setup_callbacks(self.app)
    
    def _create_overview_page(self) -> html.Div:
        """Create the main overview page"""
        
        return html.Div([
            # Welcome header
            html.Div([
                html.H1("🔬 BTC Advanced Feature Dashboard", style={
                    'textAlign': 'center',
                    'margin': '0 0 20px 0',
                    'fontSize': '36px',
                    'fontWeight': '700',
                    'background': 'linear-gradient(90deg, #00D4FF, #4ECDC4)',
                    'backgroundClip': 'text',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent'
                }),
                
                html.P("Comprehensive analysis and visualization platform for Bitcoin trading features", style={
                    'textAlign': 'center',
                    'fontSize': '18px',
                    'color': '#9CA3AF',
                    'marginBottom': '40px'
                })
            ]),
            
            # Feature categories grid
            html.Div([
                html.H2("📊 Feature Categories", style={
                    'color': 'white',
                    'marginBottom': '30px',
                    'fontSize': '24px'
                }),
                
                html.Div([
                    # RPN Features card
                    self._create_feature_card(
                        '🎯', 'RPN Features',
                        'Risk Priority Number analysis with CWT detrending, Kalman filtering, and correlation analysis',
                        '/rpn',
                        ['CWT Detrending', 'Kalman Filtering', 'EMA Analysis', 'Correlation Detection', 'Extreme Events', 'RPN Smoothing']
                    ),
                    
                    # Binning Features card
                    self._create_feature_card(
                        '📦', 'Binning Features',
                        'K-means clustering and momentum analysis for market regime classification',
                        '/binning',
                        ['K-Means Clustering', 'Momentum Analysis', 'Bin Distribution', 'Market Regimes', 'Price Coloring']
                    ),
                    
                    # Dominance Features card
                    self._create_feature_card(
                        '👑', 'Dominance Features',
                        'Market dominance detection and regime transition analysis',
                        '/dominance',
                        ['Dominance Detection', 'Regime Classification', 'Duration Analysis', 'Transition Patterns', 'Market Control']
                    ),
                    
                    # Signal Features card
                    self._create_feature_card(
                        '📡', 'Signal Features',
                        'Trading signal generation with ceiling/bottom detection and reversal analysis',
                        '/signals',
                        ['Trading Signals', 'Ceiling/Bottom Detection', 'Reversal Patterns', 'Signal Performance', 'Liquidation Analysis']
                    ),
                    
                    # Advanced Features card
                    self._create_feature_card(
                        '🚀', 'Advanced Features',
                        'Technical indicators, volatility models, and sophisticated market analysis',
                        '/advanced',
                        ['Technical Indicators', 'Volatility Models', 'KAMA', 'RSI/MACD', 'Spike Detection', 'Advanced Metrics']
                    ),
                    
                    # Statistical Analysis card
                    self._create_feature_card(
                        '📈', 'Statistical Analysis',
                        'Comprehensive statistical analysis with distribution fitting and risk metrics',
                        '/statistical',
                        ['Distribution Fitting', 'Normality Tests', 'Risk Metrics', 'Correlation Analysis', 'Executive Dashboards']
                    )
                    
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(400px, 1fr))',
                    'gap': '30px',
                    'marginBottom': '50px'
                })
            ]),
            
            # System status
            html.Div([
                html.H2("🔧 System Status", style={
                    'color': 'white',
                    'marginBottom': '20px',
                    'fontSize': '24px'
                }),
                
                html.Div(id='system-status-container', children=[
                    html.Div("Loading system status...", style={'color': '#9CA3AF'})
                ])
            ]),
            
            # Auto-refresh for system status
            dcc.Interval(
                id='overview-interval',
                interval=30000,  # Update every 30 seconds
                n_intervals=0
            )
        ])
    
    def _create_feature_card(self, icon: str, title: str, description: str, 
                           href: str, features: list) -> html.Div:
        """Create a feature category card"""
        
        return html.Div([
            html.Div([
                html.Span(icon, style={'fontSize': '32px', 'marginBottom': '15px'}),
                html.H3(title, style={
                    'color': 'white',
                    'margin': '0 0 10px 0',
                    'fontSize': '20px',
                    'fontWeight': '600'
                }),
                html.P(description, style={
                    'color': '#9CA3AF',
                    'margin': '0 0 20px 0',
                    'fontSize': '14px',
                    'lineHeight': '1.5'
                }),
                
                # Feature list
                html.Div([
                    html.H4("Features:", style={
                        'color': '#00D4FF',
                        'fontSize': '14px',
                        'marginBottom': '10px'
                    }),
                    html.Ul([
                        html.Li(feature, style={
                            'color': '#B0B0B0',
                            'fontSize': '12px',
                            'marginBottom': '5px'
                        }) for feature in features
                    ], style={'paddingLeft': '20px'})
                ], style={'marginBottom': '20px'}),
                
                # View button
                dcc.Link(
                    "View Analysis →",
                    href=href,
                    style={
                        'display': 'inline-block',
                        'padding': '10px 20px',
                        'backgroundColor': '#00D4FF',
                        'color': 'black',
                        'textDecoration': 'none',
                        'borderRadius': '6px',
                        'fontWeight': '500',
                        'fontSize': '14px',
                        'transition': 'all 0.3s ease'
                    }
                )
            ])
        ], style={
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'padding': '30px',
            'borderRadius': '15px',
            'border': '1px solid rgba(0, 212, 255, 0.2)',
            'backdropFilter': 'blur(10px)',
            'transition': 'transform 0.3s ease, box-shadow 0.3s ease',
            'cursor': 'pointer'
        }, className='chart-container')
    
    def _create_api_page(self) -> html.Div:
        """Create API documentation page"""
        
        return html.Div([
            html.H1("🔗 API Documentation", style={
                'color': 'white',
                'marginBottom': '30px',
                'fontSize': '32px'
            }),
            
            html.P("RESTful API endpoints for accessing feature data programmatically", style={
                'color': '#9CA3AF',
                'marginBottom': '40px',
                'fontSize': '16px'
            }),
            
            # API endpoints
            html.Div([
                self._create_api_endpoint_card(
                    "GET /api/features",
                    "Get feature data for specified time range",
                    "?start_date=2024-01-01&end_date=2024-01-31&hours=24"
                ),
                
                self._create_api_endpoint_card(
                    "GET /api/plots/{plot_type}",
                    "Get specific plot data in JSON format",
                    "/api/plots/price_color?hours=168"
                ),
                
                self._create_api_endpoint_card(
                    "GET /api/latest_signals",
                    "Get latest trading signals",
                    "?hours=24"
                ),
                
                self._create_api_endpoint_card(
                    "GET /api/performance",
                    "Get performance metrics",
                    "?hours=168"
                ),
                
                self._create_api_endpoint_card(
                    "GET /api/stats",
                    "Get database statistics",
                    ""
                )
            ])
        ])
    
    def _create_api_endpoint_card(self, endpoint: str, description: str, 
                                 example: str) -> html.Div:
        """Create an API endpoint documentation card"""
        
        return html.Div([
            html.H3(endpoint, style={
                'color': '#00D4FF',
                'margin': '0 0 10px 0',
                'fontSize': '18px',
                'fontFamily': 'monospace'
            }),
            html.P(description, style={
                'color': 'white',
                'margin': '0 0 15px 0',
                'fontSize': '14px'
            }),
            html.P(f"Example: {endpoint}{example}", style={
                'color': '#9CA3AF',
                'margin': '0',
                'fontSize': '12px',
                'fontFamily': 'monospace',
                'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                'padding': '8px',
                'borderRadius': '4px'
            })
        ], style={
            'backgroundColor': 'rgba(255, 255, 255, 0.05)',
            'padding': '20px',
            'borderRadius': '10px',
            'marginBottom': '20px',
            'border': '1px solid rgba(0, 212, 255, 0.2)'
        })
    
    def _setup_overview_callbacks(self):
        """Setup callbacks for overview page"""
        
        @self.app.callback(
            Output('system-status-container', 'children'),
            [Input('overview-interval', 'n_intervals')]
        )
        def update_system_status(n_intervals):
            """Update system status information"""
            
            try:
                # Get feature summary from engine
                summary = self.engine.get_feature_summary()
                
                if 'error' in summary:
                    return html.Div([
                        html.Div("❌ System Error", style={
                            'color': '#EF4444',
                            'fontWeight': '600',
                            'marginBottom': '10px'
                        }),
                        html.P(f"Error: {summary['error']}", style={'color': '#9CA3AF'})
                    ])
                
                # Create status cards
                status_cards = []
                
                # Overall status
                total_features = summary.get('total_features', 0)
                categories = summary.get('categories', {})
                available_categories = len([c for c in categories.values() if c.get('available', False)])
                
                status_cards.append(
                    html.Div([
                        html.H4("📊 Database Status", style={'color': '#00D4FF', 'margin': '0 0 10px 0'}),
                        html.P(f"✅ {available_categories}/6 feature categories available", style={'color': '#10B981', 'margin': '0'}),
                        html.P(f"📈 {total_features:,} total feature records", style={'color': '#9CA3AF', 'margin': '5px 0 0 0', 'fontSize': '14px'})
                    ], style={
                        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'border': '1px solid rgba(0, 212, 255, 0.2)',
                        'flex': '1'
                    })
                )
                
                # Category status
                for category, info in categories.items():
                    if info.get('available'):
                        icon = {'rpn': '🎯', 'bin': '📦', 'dominance': '👑', 'signals': '📡', 'advanced': '🚀'}.get(category, '📊')
                        
                        status_cards.append(
                            html.Div([
                                html.H4(f"{icon} {category.upper()}", style={'color': 'white', 'margin': '0 0 5px 0', 'fontSize': '16px'}),
                                html.P(f"{info['record_count']:,} records", style={'color': '#9CA3AF', 'margin': '0', 'fontSize': '14px'}),
                                html.P(f"Latest: {info['latest_time'][:10] if info['latest_time'] else 'N/A'}", 
                                      style={'color': '#6B7280', 'margin': '0', 'fontSize': '12px'})
                            ], style={
                                'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                                'padding': '15px',
                                'borderRadius': '8px',
                                'border': '1px solid rgba(0, 212, 255, 0.2)',
                                'flex': '1',
                                'minWidth': '150px'
                            })
                        )
                
                return html.Div([
                    html.Div(status_cards[:1], style={'marginBottom': '20px'}),  # Main status
                    html.Div(status_cards[1:], style={
                        'display': 'flex',
                        'gap': '15px',
                        'flexWrap': 'wrap'
                    })  # Category statuses
                ])
                
            except Exception as e:
                self.logger.error(f"Error updating system status: {e}")
                return html.Div([
                    html.Div("⚠️ Status Update Error", style={
                        'color': '#F59E0B',
                        'fontWeight': '600',
                        'marginBottom': '10px'
                    }),
                    html.P(f"Could not fetch system status: {str(e)}", style={'color': '#9CA3AF'})
                ])
        
        # Setup callback
        self._setup_overview_callbacks()
    
    def run_server(self, host: str = '0.0.0.0', **kwargs):
        """
        Run the dashboard server
        
        Args:
            host: Host to bind to
            **kwargs: Additional arguments for Dash server
        """
        
        self.logger.info(f"Starting unified dashboard on http://{host}:{self.port}")
        self.logger.info("Available pages:")
        self.logger.info("  📊 Overview: /")
        self.logger.info("  🎯 RPN Features: /rpn")
        self.logger.info("  📦 Binning Features: /binning")
        self.logger.info("  👑 Dominance Features: /dominance")
        self.logger.info("  📡 Signal Features: /signals")
        self.logger.info("  🚀 Advanced Features: /advanced")
        self.logger.info("  📈 Statistical Analysis: /statistical")
        self.logger.info("  🔗 API Documentation: /api")
        
        try:
            self.app.run_server(
                host=host,
                port=self.port,
                debug=self.debug,
                dev_tools_hot_reload=False,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")
            raise


def create_unified_dashboard(db_config: Optional[Dict] = None,
                           port: int = 8050,
                           debug: bool = False) -> UnifiedDashboard:
    """
    Create and return a unified dashboard instance
    
    Args:
        db_config: Database configuration
        port: Port for web server
        debug: Enable debug mode
        
    Returns:
        UnifiedDashboard instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        dashboard = UnifiedDashboard(db_config, port, debug)
        return dashboard
        
    except Exception as e:
        print(f"Error creating unified dashboard: {e}")
        raise


if __name__ == "__main__":
    # Run the unified dashboard
    dashboard = create_unified_dashboard(port=8050, debug=True)
    dashboard.run_server(debug=True)