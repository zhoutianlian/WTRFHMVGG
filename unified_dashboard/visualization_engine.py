"""
Unified Visualization Engine

Integrates all visualization functions from different modules into a single engine
that can generate any chart for any feature category.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Add paths for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'feature_engineer'))
sys.path.append(str(current_dir / 'dashboard'))

# Import all visualization modules
try:
    # Advanced visualizations
    from feature_engineer.advanced_visualization.rpn_visualizations import RPNVisualizer
    from feature_engineer.advanced_visualization.binning_visualizations import BinningVisualizer
    from feature_engineer.advanced_visualization.dominance_visualizations import DominanceVisualizer
    from feature_engineer.advanced_visualization.signal_visualizations import SignalVisualizer
    
    # Dashboard visualizations
    from dashboard.plotly_visualizer import PlotlyVisualizer
    
    # Statistical analysis
    from feature_engineer.advanced_features.statistical_analyzer import FeatureStatisticalAnalyzer
    
    # Database components
    from feature_engineer.db.database_manager import DatabaseManager
    from feature_engineer.data_reader.reader import BTCDataReader
    
except ImportError as e:
    print(f"Warning: Could not import some visualization modules: {e}")


class VisualizationEngine:
    """
    Unified engine that can generate any visualization from any module
    
    Provides a single interface to access all visualization functions:
    - RPN features (CWT, Kalman, EMA, correlation analysis)
    - Binning features (K-means clustering, momentum analysis)  
    - Dominance features (market dominance, regime classification)
    - Signal features (trading signals, ceiling/bottom detection)
    - Advanced features (technical indicators, volatility models)
    - Statistical analysis (distribution fitting, normality tests)
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize database components
        self.db = DatabaseManager(db_config)
        self.reader = BTCDataReader(self.db)
        
        # Initialize statistical analyzer
        self.stats_analyzer = FeatureStatisticalAnalyzer(self.db, self.logger)
        
        # Cache for visualizers to avoid recreating them
        self._visualizer_cache = {}
        
        self.logger.info("Visualization engine initialized successfully")
    
    def get_data(self, start_time: Optional[datetime] = None, 
                 end_time: Optional[datetime] = None,
                 feature_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get unified data for all feature categories
        
        Args:
            start_time: Start of data window
            end_time: End of data window  
            feature_categories: Specific categories to include
            
        Returns:
            Combined DataFrame with all feature data
        """
        try:
            # Get combined features from all tables
            df = self.reader.get_combined_features(start_time, end_time)
            
            if df.empty:
                self.logger.warning(f"No data found for time range {start_time} to {end_time}")
                return pd.DataFrame()
            
            # Filter by categories if specified
            if feature_categories:
                # Map categories to column patterns
                category_patterns = {
                    'rpn': ['risk_priority_number', 'lld_cwt_kf', 'fll_cwt_kf', 'fsl_cwt_kf', 
                           'diff_ls_cwt_kf', 'lld_cwt_kf_smooth', 'corr_case', 'is_rpn_extreme'],
                    'bin': ['bin_index', 'bin_momentum', 'bin_turn', 'bin_mm_direction'],
                    'dominance': ['dominance', 'dominance_duration', 'dominance_time', 
                                 'dominance_class', 'is_keep', 'is_strengthen'],
                    'signals': ['hit_ceiling_bottom', 'reverse_ceiling_bottom', 
                               'signal_l1', 'signal_l2', 'signal_l3', 'signal_class'],
                    'advanced': ['kama', 'spike_detection', 'roc', 'velocity', 'acceleration',
                                'rsi', 'stoch_k', 'stoch_d', 'macd', 'kalman_slope']
                }
                
                # Keep base columns
                keep_columns = ['time', 'price']
                
                # Add category-specific columns
                for category in feature_categories:
                    if category in category_patterns:
                        for pattern in category_patterns[category]:
                            if pattern in df.columns:
                                keep_columns.append(pattern)
                
                df = df[keep_columns]
            
            # Ensure time column is datetime and sorted
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data: {e}")
            return pd.DataFrame()
    
    def get_visualizer(self, category: str, df: pd.DataFrame) -> Any:
        """
        Get appropriate visualizer for feature category
        
        Args:
            category: Feature category ('rpn', 'bin', 'dominance', 'signals', 'advanced', 'statistical')
            df: DataFrame with feature data
            
        Returns:
            Visualizer instance for the category
        """
        cache_key = f"{category}_{len(df)}"
        
        if cache_key in self._visualizer_cache:
            return self._visualizer_cache[cache_key]
        
        try:
            if category == 'rpn':
                visualizer = RPNVisualizer(df)
            elif category == 'bin' or category == 'binning':
                visualizer = BinningVisualizer(df)
            elif category == 'dominance':
                visualizer = DominanceVisualizer(df)
            elif category == 'signals':
                visualizer = SignalVisualizer(df)
            elif category == 'backend' or category == 'plotly':
                visualizer = PlotlyVisualizer(df)
            elif category == 'statistical':
                # For statistical analysis, return the analyzer
                visualizer = self.stats_analyzer
            else:
                # Default to PlotlyVisualizer for unknown categories
                visualizer = PlotlyVisualizer(df)
            
            self._visualizer_cache[cache_key] = visualizer
            return visualizer
            
        except Exception as e:
            self.logger.error(f"Error creating visualizer for {category}: {e}")
            # Fallback to basic PlotlyVisualizer
            visualizer = PlotlyVisualizer(df)
            self._visualizer_cache[cache_key] = visualizer
            return visualizer
    
    def create_chart(self, chart_type: str, category: str, df: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Create any chart for any feature category
        
        Args:
            chart_type: Type of chart to create
            category: Feature category
            df: Data for visualization
            **kwargs: Additional parameters for chart creation
            
        Returns:
            Plotly Figure object
        """
        try:
            visualizer = self.get_visualizer(category, df)
            
            # Map chart types to visualizer methods
            chart_methods = self._get_chart_methods(category)
            
            if chart_type not in chart_methods:
                self.logger.warning(f"Unknown chart type '{chart_type}' for category '{category}'")
                return self._create_fallback_chart(df, chart_type)
            
            method_name = chart_methods[chart_type]
            
            if hasattr(visualizer, method_name):
                method = getattr(visualizer, method_name)
                fig = method(**kwargs)
                
                # Convert dict to Figure if needed (for PlotlyVisualizer)
                if isinstance(fig, dict):
                    fig = go.Figure(fig)
                
                return fig
            else:
                self.logger.warning(f"Method '{method_name}' not found in {category} visualizer")
                return self._create_fallback_chart(df, chart_type)
                
        except Exception as e:
            self.logger.error(f"Error creating {chart_type} chart for {category}: {e}")
            return self._create_fallback_chart(df, chart_type)
    
    def _get_chart_methods(self, category: str) -> Dict[str, str]:
        """Get mapping of chart types to method names for each category"""
        
        method_maps = {
            'rpn': {
                'dominance_plot': 'create_rpn_dominance_plot',
                'cwt_analysis': 'create_cwt_analysis_plot',
                'kalman_analysis': 'create_kalman_analysis_plot',
                'correlation_analysis': 'create_correlation_analysis_plot',
                'extreme_conditions': 'create_extreme_conditions_plot',
                'ema_analysis': 'create_ema_analysis_plot',
                'rpn_overview': 'create_rpn_overview_dashboard'
            },
            'bin': {
                'price_color': 'create_price_color_plot',
                'bin_analysis': 'create_bin_analysis_dashboard',
                'clustering_analysis': 'create_clustering_analysis_plot',
                'momentum_patterns': 'create_momentum_patterns_plot',
                'market_regime': 'create_market_regime_plot',
                'bin_distribution': 'create_bin_distribution_plot',
                'bin_overview': 'create_binning_overview_dashboard'
            },
            'binning': {  # Alias for bin
                'price_color': 'create_price_color_plot',
                'bin_analysis': 'create_bin_analysis_dashboard',
                'clustering_analysis': 'create_clustering_analysis_plot',
                'momentum_patterns': 'create_momentum_patterns_plot',
                'market_regime': 'create_market_regime_plot'
            },
            'dominance': {
                'dominance_status': 'create_dominance_status_plot',
                'market_regime': 'create_market_regime_dashboard',
                'dominance_patterns': 'create_dominance_patterns_plot',
                'dominance_correlation': 'create_dominance_correlation_plot',
                'regime_transitions': 'create_regime_transition_analysis',
                'dominance_overview': 'create_dominance_overview_dashboard'
            },
            'signals': {
                'signals_plot': 'create_signals_plot',
                'liquidations_analysis': 'create_liquidations_analysis_plot',
                'trading_signals': 'create_trading_signals_dashboard',
                'ceiling_bottom': 'create_ceiling_bottom_analysis',
                'reversal_signals': 'create_reversal_signals_analysis',
                'signal_performance': 'create_signal_performance_analysis',
                'signals_overview': 'create_signals_overview_dashboard'
            },
            'backend': {
                'price_color': 'create_price_color_plot',
                'rpn_dominance': 'create_rpn_dominance_plot',
                'fll_fsl': 'create_fll_fsl_plot',
                'dominance_status': 'create_dominance_status_plot',
                'signals': 'create_signals_plot'
            },
            'plotly': {  # Alias for backend
                'price_color': 'create_price_color_plot',
                'rpn_dominance': 'create_rpn_dominance_plot',
                'fll_fsl': 'create_fll_fsl_plot',
                'dominance_status': 'create_dominance_status_plot',
                'signals': 'create_signals_plot'
            }
        }
        
        return method_maps.get(category, {})
    
    def _create_fallback_chart(self, df: pd.DataFrame, chart_type: str) -> go.Figure:
        """Create a fallback chart when specific chart type is not available"""
        
        fig = go.Figure()
        
        if 'price' in df.columns and 'time' in df.columns:
            # Default price chart
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['price'],
                mode='lines',
                name='BTC Price',
                line=dict(color='#00D4FF', width=2)
            ))
            
            fig.update_layout(
                title=f'Fallback Chart: {chart_type}',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                template='plotly_dark',
                height=400
            )
        else:
            # No data chart
            fig.add_annotation(
                text=f"No data available for {chart_type}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="white")
            )
            
            fig.update_layout(
                title=f'No Data: {chart_type}',
                template='plotly_dark',
                height=400
            )
        
        return fig
    
    def get_available_charts(self, category: str) -> List[str]:
        """Get list of available chart types for a feature category"""
        chart_methods = self._get_chart_methods(category)
        return list(chart_methods.keys())
    
    def get_statistical_analysis(self, start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               feature_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistical analysis for features
        
        Args:
            start_time: Start of analysis window
            end_time: End of analysis window
            feature_categories: Categories to analyze
            
        Returns:
            Statistical analysis results
        """
        try:
            return self.stats_analyzer.analyze_all_features(
                start_time, end_time, feature_categories
            )
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            return {'error': str(e)}
    
    def create_comparison_dashboard(self, categories: List[str], 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> go.Figure:
        """
        Create a comparison dashboard showing multiple feature categories
        
        Args:
            categories: List of feature categories to compare
            start_time: Start of data window
            end_time: End of data window
            
        Returns:
            Combined dashboard figure
        """
        try:
            # Get data for all categories
            df = self.get_data(start_time, end_time, categories)
            
            if df.empty:
                return self._create_fallback_chart(df, "Comparison Dashboard")
            
            # Create subplot layout
            rows = min(len(categories), 4)  # Max 4 rows
            
            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[f'{cat.upper()} Features' for cat in categories[:rows]]
            )
            
            # Add key chart for each category
            for i, category in enumerate(categories[:rows], 1):
                try:
                    visualizer = self.get_visualizer(category, df)
                    
                    # Get the main chart for each category
                    if category == 'rpn':
                        if hasattr(visualizer, 'create_rpn_dominance_plot'):
                            sub_fig = visualizer.create_rpn_dominance_plot()
                    elif category in ['bin', 'binning']:
                        if hasattr(visualizer, 'create_price_color_plot'):
                            sub_fig = visualizer.create_price_color_plot()
                    elif category == 'dominance':
                        if hasattr(visualizer, 'create_dominance_status_plot'):
                            sub_fig = visualizer.create_dominance_status_plot()
                    elif category == 'signals':
                        if hasattr(visualizer, 'create_signals_plot'):
                            sub_fig = visualizer.create_signals_plot()
                    else:
                        # Fallback to price chart
                        fig.add_trace(
                            go.Scatter(
                                x=df['time'],
                                y=df['price'],
                                mode='lines',
                                name=f'{category.upper()} Price',
                                line=dict(width=2)
                            ),
                            row=i, col=1
                        )
                        continue
                    
                    # Extract traces from sub_fig if created successfully
                    if 'sub_fig' in locals() and hasattr(sub_fig, 'data'):
                        for trace in sub_fig.data[:2]:  # Limit to 2 traces per category
                            fig.add_trace(trace, row=i, col=1)
                    
                except Exception as e:
                    self.logger.warning(f"Error adding {category} to comparison: {e}")
                    # Add fallback trace
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'],
                            y=df['price'],
                            mode='lines',
                            name=f'{category.upper()} (Fallback)',
                            line=dict(width=1, dash='dash')
                        ),
                        row=i, col=1
                    )
            
            fig.update_layout(
                title='Feature Categories Comparison Dashboard',
                template='plotly_dark',
                height=300 * rows,
                showlegend=True
            )
            
            # Update x-axis for bottom subplot only
            fig.update_xaxes(title_text='Time', row=rows, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating comparison dashboard: {e}")
            return self._create_fallback_chart(pd.DataFrame(), "Comparison Dashboard Error")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all available features and their status"""
        
        try:
            summary = {
                'categories': {},
                'total_features': 0,
                'database_status': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check each feature table
            feature_tables = {
                'rpn': 'btc_features_rpn',
                'bin': 'btc_features_bin', 
                'dominance': 'btc_features_dominance',
                'signals': 'btc_features_signals',
                'advanced': 'btc_features_advanced'
            }
            
            for category, table in feature_tables.items():
                try:
                    stats = self.reader.get_feature_stats(table)
                    
                    if 'error' not in stats:
                        summary['categories'][category] = {
                            'available': True,
                            'record_count': stats['record_count'],
                            'latest_time': stats['max_time'].isoformat() if stats['max_time'] else None,
                            'oldest_time': stats['min_time'].isoformat() if stats['min_time'] else None
                        }
                        summary['total_features'] += stats['record_count']
                    else:
                        summary['categories'][category] = {
                            'available': False,
                            'error': stats['error']
                        }
                        
                except Exception as e:
                    summary['categories'][category] = {
                        'available': False,
                        'error': str(e)
                    }
            
            # Available chart types
            summary['available_charts'] = {}
            for category in feature_tables.keys():
                summary['available_charts'][category] = self.get_available_charts(category)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting feature summary: {e}")
            return {'error': str(e)}


def create_visualization_engine(db_config: Optional[Dict] = None) -> VisualizationEngine:
    """
    Convenience function to create visualization engine
    
    Args:
        db_config: Database configuration
        
    Returns:
        VisualizationEngine instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return VisualizationEngine(db_config)


if __name__ == "__main__":
    # Example usage
    engine = create_visualization_engine()
    
    # Get feature summary
    summary = engine.get_feature_summary()
    print(f"Available categories: {list(summary['categories'].keys())}")
    
    # Create sample comparison dashboard
    df = engine.get_data()
    if not df.empty:
        fig = engine.create_comparison_dashboard(['rpn', 'bin', 'dominance'])
        print("Comparison dashboard created successfully")