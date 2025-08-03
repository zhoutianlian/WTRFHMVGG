"""
Statistical Analysis Module for Advanced Features

This module integrates statistic_cis.py functionality to analyze all features
in the advanced_features system, reading data from the database and providing
comprehensive statistical analysis with beautiful visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import logging
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database_manager import DatabaseManager

# Import statistic_cis functions
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from statistic_cis import (
        analyze_time_series, fit_t_distribution, fit_gaussian_mixture,
        test_normality, create_beautiful_distribution_plot,
        create_executive_dashboard, create_comparison_plot
    )
except ImportError as e:
    print(f"Warning: Could not import statistic_cis functions: {e}")
    print("Some statistical analysis features may not be available")


class FeatureStatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for all advanced features
    
    Integrates statistic_cis.py functionality to analyze:
    - RPN features (CWT, Kalman, EMA, correlation analysis)
    - Binning features (K-means clustering, momentum)
    - Dominance features (market dominance, regime classification)
    - Signal features (trading signals, ceiling/bottom detection)
    - Advanced features (KAMA, spikes, volatility, technical indicators)
    """
    
    def __init__(self, db_manager: DatabaseManager, logger: Optional[logging.Logger] = None):
        self.db = db_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Feature tables and their key columns
        self.feature_tables = {
            'rpn': {
                'table': 'btc_features_rpn',
                'columns': [
                    'risk_priority_number', 'lld_cwt_kf', 'fll_cwt_kf', 'fsl_cwt_kf',
                    'diff_ls_cwt_kf', 'lld_cwt_kf_smooth', 'diff_ls_smooth',
                    'delta_fll', 'delta_fsl', 'is_rpn_extreme'
                ],
                'description': 'Risk Priority Number and CWT/Kalman features'
            },
            'bin': {
                'table': 'btc_features_bin',
                'columns': [
                    'bin_index', 'bin_momentum', 'bin_turn', 'bin_mm_direction'
                ],
                'description': 'K-means binning and momentum features'
            },
            'dominance': {
                'table': 'btc_features_dominance',
                'columns': [
                    'dominance', 'dominance_last', 'dominance_duration',
                    'dominance_duration_total', 'dominance_time', 'is_keep', 'is_strengthen'
                ],
                'description': 'Market dominance and regime features'
            },
            'signals': {
                'table': 'btc_features_signals',
                'columns': [
                    'hit_ceiling_bottom', 'reverse_ceiling_bottom',
                    'signal_l1', 'signal_l2', 'signal_l3'
                ],
                'description': 'Trading signals and ceiling/bottom detection'
            },
            'advanced': {
                'table': 'btc_features_advanced',
                'columns': [
                    'kama', 'spike_detection', 'roc', 'velocity', 'acceleration',
                    'rsi', 'stoch_k', 'stoch_d', 'macd', 'kalman_slope',
                    'skewness', 'kurtosis', 'vol_adaptive', 'vol_garch'
                ],
                'description': 'Advanced technical and statistical features'
            }
        }
        
        self.logger.info(f"Initialized statistical analyzer for {len(self.feature_tables)} feature categories")
    
    def analyze_all_features(self, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           feature_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze all features across all categories with comprehensive statistics
        
        Args:
            start_time: Start of analysis window
            end_time: End of analysis window
            feature_categories: List of categories to analyze (default: all)
            
        Returns:
            Dictionary with analysis results for each feature category
        """
        if feature_categories is None:
            feature_categories = list(self.feature_tables.keys())
        
        self.logger.info(f"Starting statistical analysis for categories: {feature_categories}")
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'time_window': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            },
            'categories': {}
        }
        
        for category in feature_categories:
            if category not in self.feature_tables:
                self.logger.warning(f"Unknown feature category: {category}")
                continue
            
            try:
                self.logger.info(f"Analyzing {category} features...")
                category_results = self._analyze_feature_category(category, start_time, end_time)
                results['categories'][category] = category_results
                
                self.logger.info(f"Completed analysis for {category}: {len(category_results.get('features', {}))} features analyzed")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {category} features: {e}")
                results['categories'][category] = {'error': str(e)}
        
        # Generate executive summary
        results['executive_summary'] = self._generate_executive_summary(results)
        
        self.logger.info(f"Statistical analysis complete for {len(results['categories'])} categories")
        return results
    
    def _analyze_feature_category(self, category: str, start_time: Optional[datetime],
                                 end_time: Optional[datetime]) -> Dict[str, Any]:
        """Analyze all features in a specific category"""
        
        table_info = self.feature_tables[category]
        table_name = table_info['table']
        
        # Get data from database
        if start_time or end_time:
            df = self.db.get_data_range(table_name, start_time, end_time)
        else:
            df = self.db.read_data(f"SELECT * FROM {table_name} ORDER BY time")
        
        if df.empty:
            return {'error': f'No data available in {table_name}'}
        
        category_results = {
            'description': table_info['description'],
            'data_summary': {
                'records': len(df),
                'time_range': {
                    'start': df['time'].min().isoformat() if 'time' in df.columns else None,
                    'end': df['time'].max().isoformat() if 'time' in df.columns else None
                },
                'columns': df.columns.tolist()
            },
            'features': {}
        }
        
        # Analyze each numeric feature in the category
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove time-based columns from analysis
        analysis_columns = [col for col in numeric_columns 
                          if col not in ['time'] and col in table_info['columns']]
        
        for feature_name in analysis_columns:
            if feature_name not in df.columns:
                continue
                
            try:
                feature_data = df[feature_name].dropna()
                
                if len(feature_data) < 10:  # Need minimum data points
                    category_results['features'][feature_name] = {
                        'error': 'Insufficient data points for analysis'
                    }
                    continue
                
                # Create time series for analysis
                if 'time' in df.columns:
                    feature_df = df[['time', feature_name]].dropna()
                    feature_df = feature_df.rename(columns={'time': 'time', feature_name: 'value'})
                else:
                    # Create synthetic time series
                    feature_df = pd.DataFrame({
                        'time': pd.date_range(start='2020-01-01', periods=len(feature_data), freq='H'),
                        'value': feature_data.values
                    })
                
                # Perform comprehensive analysis using statistic_cis functions
                feature_analysis = self._analyze_single_feature(feature_df, feature_name, category)
                category_results['features'][feature_name] = feature_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing feature {feature_name}: {e}")
                category_results['features'][feature_name] = {'error': str(e)}
        
        return category_results
    
    def _analyze_single_feature(self, df: pd.DataFrame, feature_name: str, category: str) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on a single feature"""
        
        analysis_results = {
            'basic_stats': self._calculate_basic_statistics(df['value']),
            'distribution_analysis': {},
            'normality_tests': {},
            'time_series_analysis': {},
            'risk_metrics': {},
            'visualization_paths': []
        }
        
        # Use statistic_cis functions if available
        try:
            # Main time series analysis
            ts_analysis = analyze_time_series(df, time_col='time', value_col='value')
            if ts_analysis:
                analysis_results['time_series_analysis'] = ts_analysis
            
            # Distribution fitting
            t_dist_results = fit_t_distribution(df['value'].values)
            if t_dist_results:
                analysis_results['distribution_analysis']['t_distribution'] = t_dist_results
            
            gmm_results = fit_gaussian_mixture(df['value'].values)
            if gmm_results:
                analysis_results['distribution_analysis']['gaussian_mixture'] = gmm_results
            
            # Normality tests
            normality_results = test_normality(df['value'].values)
            if normality_results:
                analysis_results['normality_tests'] = normality_results
            
        except Exception as e:
            self.logger.warning(f"Could not perform advanced analysis for {feature_name}: {e}")
        
        # Calculate risk metrics
        analysis_results['risk_metrics'] = self._calculate_risk_metrics(df['value'])
        
        # Generate visualizations
        analysis_results['visualization_paths'] = self._create_feature_visualizations(
            df, feature_name, category, analysis_results
        )
        
        return analysis_results
    
    def _calculate_basic_statistics(self, data: pd.Series) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        return {
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(data.median()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'skewness': float(data.skew()),
            'kurtosis': float(data.kurtosis()),
            'variance': float(data.var()),
            'cv': float(data.std() / data.mean()) if data.mean() != 0 else np.inf
        }
    
    def _calculate_risk_metrics(self, data: pd.Series) -> Dict[str, float]:
        """Calculate financial risk metrics"""
        returns = data.pct_change().dropna()
        
        metrics = {}
        
        if len(returns) > 1:
            metrics.update({
                'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': float((data / data.cummax() - 1).min()),
                'var_95': float(returns.quantile(0.05)),
                'var_99': float(returns.quantile(0.01)),
                'expected_shortfall_95': float(returns[returns <= returns.quantile(0.05)].mean()),
                'expected_shortfall_99': float(returns[returns <= returns.quantile(0.01)].mean())
            })
        
        return metrics
    
    def _create_feature_visualizations(self, df: pd.DataFrame, feature_name: str, 
                                     category: str, analysis_results: Dict) -> List[str]:
        """Create comprehensive visualizations for a feature"""
        
        visualization_paths = []
        
        # Create output directory
        viz_dir = Path(f"statistical_analysis/{category}")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Time series plot
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df['time'],
                y=df['value'],
                mode='lines',
                name=feature_name,
                line=dict(color='#00D4FF', width=2)
            ))
            
            fig_ts.update_layout(
                title=f'{feature_name.replace("_", " ").title()} Time Series ({category.upper()})',
                xaxis_title='Time',
                yaxis_title='Value',
                template='plotly_dark',
                height=400
            )
            
            ts_path = viz_dir / f"{feature_name}_timeseries.html"
            fig_ts.write_html(str(ts_path))
            visualization_paths.append(str(ts_path))
            
            # 2. Distribution plot using statistic_cis if available
            try:
                dist_fig = create_beautiful_distribution_plot(
                    df['value'].values, 
                    title=f'{feature_name.replace("_", " ").title()} Distribution'
                )
                if dist_fig:
                    dist_path = viz_dir / f"{feature_name}_distribution.html"
                    dist_fig.write_html(str(dist_path))
                    visualization_paths.append(str(dist_path))
            except:
                # Fallback distribution plot
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=df['value'],
                    nbinsx=50,
                    name='Histogram',
                    marker_color='#00D4FF',
                    opacity=0.7
                ))
                
                fig_dist.update_layout(
                    title=f'{feature_name.replace("_", " ").title()} Distribution',
                    template='plotly_dark',
                    height=400
                )
                
                dist_path = viz_dir / f"{feature_name}_distribution_simple.html"
                fig_dist.write_html(str(dist_path))
                visualization_paths.append(str(dist_path))
            
            # 3. Statistics summary plot
            stats = analysis_results['basic_stats']
            fig_stats = go.Figure()
            
            fig_stats.add_trace(go.Bar(
                x=['Mean', 'Median', 'Std', 'Skewness', 'Kurtosis'],
                y=[stats['mean'], stats['median'], stats['std'], stats['skewness'], stats['kurtosis']],
                marker_color=['#00D4FF', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                text=[f'{v:.4f}' for v in [stats['mean'], stats['median'], stats['std'], 
                                         stats['skewness'], stats['kurtosis']]],
                textposition='auto'
            ))
            
            fig_stats.update_layout(
                title=f'{feature_name.replace("_", " ").title()} Statistical Summary',
                template='plotly_dark',
                height=400
            )
            
            stats_path = viz_dir / f"{feature_name}_statistics.html"
            fig_stats.write_html(str(stats_path))
            visualization_paths.append(str(stats_path))
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations for {feature_name}: {e}")
        
        return visualization_paths
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of all analyses"""
        
        summary = {
            'total_categories': len(results['categories']),
            'total_features': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'key_insights': [],
            'recommendations': []
        }
        
        all_risk_metrics = []
        all_basic_stats = []
        
        for category, category_data in results['categories'].items():
            if 'error' in category_data:
                summary['failed_analyses'] += 1
                continue
            
            features = category_data.get('features', {})
            summary['total_features'] += len(features)
            
            for feature_name, feature_data in features.items():
                if 'error' in feature_data:
                    summary['failed_analyses'] += 1
                else:
                    summary['successful_analyses'] += 1
                    
                    # Collect metrics for summary
                    if 'basic_stats' in feature_data:
                        all_basic_stats.append({
                            'category': category,
                            'feature': feature_name,
                            **feature_data['basic_stats']
                        })
                    
                    if 'risk_metrics' in feature_data:
                        all_risk_metrics.append({
                            'category': category,
                            'feature': feature_name,
                            **feature_data['risk_metrics']
                        })
        
        # Generate insights
        if all_basic_stats:
            # Find most volatile features
            volatile_features = sorted(all_basic_stats, key=lambda x: x.get('cv', 0), reverse=True)[:3]
            summary['key_insights'].append({
                'type': 'highest_volatility',
                'features': [f"{f['category']}.{f['feature']}" for f in volatile_features],
                'values': [f.get('cv', 0) for f in volatile_features]
            })
            
            # Find most skewed features
            skewed_features = sorted(all_basic_stats, key=lambda x: abs(x.get('skewness', 0)), reverse=True)[:3]
            summary['key_insights'].append({
                'type': 'highest_skewness',
                'features': [f"{f['category']}.{f['feature']}" for f in skewed_features],
                'values': [f.get('skewness', 0) for f in skewed_features]
            })
        
        if all_risk_metrics:
            # Find highest risk features
            risky_features = sorted(all_risk_metrics, key=lambda x: x.get('volatility', 0), reverse=True)[:3]
            summary['key_insights'].append({
                'type': 'highest_risk',
                'features': [f"{f['category']}.{f['feature']}" for f in risky_features],
                'values': [f.get('volatility', 0) for f in risky_features]
            })
        
        # Generate recommendations
        summary['recommendations'] = [
            "Monitor high-volatility features for trading signals",
            "Consider risk management for features with high VaR",
            "Investigate non-normal distributions for regime detection",
            "Use correlation analysis for feature selection",
            "Implement dynamic thresholds based on statistical properties"
        ]
        
        return summary
    
    def create_comparison_dashboard(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Create a comprehensive dashboard comparing all features"""
        
        if output_path is None:
            output_path = f"statistical_analysis/feature_comparison_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Create comparison plots using statistic_cis if available
        try:
            dashboard_fig = create_executive_dashboard(results)
            if dashboard_fig:
                dashboard_fig.write_html(output_path)
                self.logger.info(f"Created comparison dashboard: {output_path}")
                return output_path
        except:
            self.logger.warning("Could not create executive dashboard using statistic_cis, creating simple dashboard")
        
        # Fallback: create simple comparison dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Volatility Comparison', 'Skewness Distribution', 
                          'Risk Metrics', 'Feature Counts'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract data for plots
        volatility_data = []
        skewness_data = []
        categories = []
        
        for category, category_data in results['categories'].items():
            if 'error' in category_data:
                continue
            
            features = category_data.get('features', {})
            for feature_name, feature_data in features.items():
                if 'error' not in feature_data and 'basic_stats' in feature_data:
                    stats = feature_data['basic_stats']
                    volatility_data.append(stats.get('cv', 0))
                    skewness_data.append(stats.get('skewness', 0))
                    categories.append(f"{category}.{feature_name}")
        
        # Add traces
        if volatility_data:
            fig.add_trace(
                go.Bar(x=categories[:10], y=volatility_data[:10], name='CV'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=skewness_data, name='Skewness'),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Feature Statistical Analysis Dashboard',
            template='plotly_dark',
            height=800
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        
        self.logger.info(f"Created simple comparison dashboard: {output_path}")
        return output_path
    
    def export_analysis_results(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Export analysis results to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            import json
            output_path = f"statistical_analysis/feature_analysis_{timestamp}.json"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format == 'csv':
            # Export flattened results as CSV
            output_path = f"statistical_analysis/feature_analysis_{timestamp}.csv"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            rows = []
            for category, category_data in results['categories'].items():
                if 'error' in category_data:
                    continue
                
                features = category_data.get('features', {})
                for feature_name, feature_data in features.items():
                    if 'error' not in feature_data:
                        row = {
                            'category': category,
                            'feature': feature_name,
                            'description': self.feature_tables[category]['description']
                        }
                        
                        # Add basic stats
                        if 'basic_stats' in feature_data:
                            row.update({f'basic_{k}': v for k, v in feature_data['basic_stats'].items()})
                        
                        # Add risk metrics
                        if 'risk_metrics' in feature_data:
                            row.update({f'risk_{k}': v for k, v in feature_data['risk_metrics'].items()})
                        
                        rows.append(row)
            
            if rows:
                df_export = pd.DataFrame(rows)
                df_export.to_csv(output_path, index=False)
        
        self.logger.info(f"Exported analysis results to: {output_path}")
        return output_path


def run_statistical_analysis(db_config: Optional[dict] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           feature_categories: Optional[List[str]] = None,
                           export_format: str = 'json') -> Dict[str, Any]:
    """
    Convenience function to run complete statistical analysis
    
    Args:
        db_config: Database configuration
        start_time: Start of analysis window
        end_time: End of analysis window
        feature_categories: Categories to analyze
        export_format: Export format ('json' or 'csv')
        
    Returns:
        Analysis results dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_config)
        analyzer = FeatureStatisticalAnalyzer(db, logger)
        
        # Run analysis
        results = analyzer.analyze_all_features(start_time, end_time, feature_categories)
        
        # Export results
        export_path = analyzer.export_analysis_results(results, export_format)
        
        # Create dashboard
        dashboard_path = analyzer.create_comparison_dashboard(results)
        
        logger.info(f"Statistical analysis complete:")
        logger.info(f"- Results exported to: {export_path}")
        logger.info(f"- Dashboard created: {dashboard_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    results = run_statistical_analysis()