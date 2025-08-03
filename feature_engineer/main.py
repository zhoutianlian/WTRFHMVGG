#!/usr/bin/env python3
"""
Main orchestrator for BTC Feature Engineering Pipeline

This script provides a command-line interface and scheduling capabilities
for the complete PostgreSQL-based feature engineering pipeline.

Usage Examples:
    # Run full pipeline
    python main.py --full-pipeline
    
    # Download only
    python main.py --download --full-refresh
    
    # Process features only
    python main.py --features
    
    # Export training data
    python main.py --export --output training_data.csv
    
    # Get pipeline status
    python main.py --status
    
    # Run scheduled updates
    python main.py --schedule --interval 3600  # Every hour
    
    # Run statistical analysis
    python main.py --analyze --categories rpn bin dominance
    
    # Start web dashboard
    python main.py --web-dashboard --port 8051
    
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.database_manager import DatabaseManager
from data_download.downloader import BTCDataDownloader
from preprocessing.preprocessor import DataPreprocessor
from advanced_features.pipeline import FeatureEngineer
from data_reader.reader import BTCDataReader
from config.config_manager import ConfigManager


class BTCFeaturePipeline:
    """Main orchestrator for BTC feature engineering pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        try:
            self.db = DatabaseManager(self.config.get('database'))
            self.downloader = BTCDataDownloader(self.db, self.logger)
            self.preprocessor = DataPreprocessor(self.db, self.logger)
            self.feature_engineer = FeatureEngineer(self.db, self.logger)
            self.data_reader = BTCDataReader(self.db, self.logger)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('file', 'btc_pipeline.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_database(self):
        """Initialize database schema"""
        try:
            self.logger.info("Initializing database schema...")
            self.db.create_tables()
            self.logger.info("Database schema initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    def run_download(self, full_refresh: bool = False) -> Dict[str, int]:
        """
        Run data download step
        
        Args:
            full_refresh: If True, download all historical data
            
        Returns:
            Dictionary with download results
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING DATA DOWNLOAD")
            self.logger.info("=" * 60)
            
            results = self.downloader.download_and_store_btc_data(full_refresh=full_refresh)
            
            total_records = sum(results.values())
            self.logger.info(f"Data download completed: {total_records} total records")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data download failed: {e}")
            raise
    
    def run_preprocessing(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, int]:
        """
        Run data preprocessing step
        
        Args:
            start_time: Start of processing window
            end_time: End of processing window
            
        Returns:
            Dictionary with preprocessing results
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING DATA PREPROCESSING")
            self.logger.info("=" * 60)
            
            results = self.preprocessor.run_preprocessing(start_time, end_time)
            
            self.logger.info(f"Data preprocessing completed: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def run_feature_engineering(self, start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Dict[str, int]:
        """
        Run feature engineering step
        
        Args:
            start_time: Start of processing window
            end_time: End of processing window
            
        Returns:
            Dictionary with feature engineering results
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING FEATURE ENGINEERING")
            self.logger.info("=" * 60)
            
            results = self.feature_engineer.run_feature_engineering(start_time, end_time)
            
            self.logger.info(f"Feature engineering completed: {results}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise
    
    def run_full_pipeline(self, full_refresh: bool = False) -> Dict[str, Any]:
        """
        Run complete pipeline: download -> preprocess -> engineer features
        
        Args:
            full_refresh: If True, download all historical data
            
        Returns:
            Dictionary with all pipeline results
        """
        pipeline_start = datetime.now()
        results = {
            'start_time': pipeline_start.isoformat(),
            'status': 'running',
            'steps': {}
        }
        
        try:
            self.logger.info("🚀 STARTING FULL BTC FEATURE ENGINEERING PIPELINE")
            self.logger.info("=" * 80)
            
            # Step 1: Download data
            download_results = self.run_download(full_refresh=full_refresh)
            results['steps']['download'] = download_results
            
            # Step 2: Preprocess data
            preprocessing_results = self.run_preprocessing()
            results['steps']['preprocessing'] = preprocessing_results
            
            # Step 3: Engineer features
            feature_results = self.run_feature_engineering()
            results['steps']['feature_engineering'] = feature_results
            
            # Pipeline completion
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            results['status'] = 'completed'
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = duration
            
            self.logger.info("=" * 80)
            self.logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f} seconds")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = pipeline_end.isoformat()
            results['duration_seconds'] = duration
            
            self.logger.error(f"❌ PIPELINE FAILED after {duration:.2f} seconds: {e}")
            raise
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'database': {},
                'data_summary': {}
            }
            
            # Database status
            for table in ['btc_price_data', 'btc_liquidations', 'btc_processed_data', 
                         'btc_features_rpn', 'btc_features_bin', 'btc_features_dominance', 'btc_features_signals']:
                try:
                    latest = self.db.get_latest_timestamp(table)
                    count_df = self.db.read_data(f"SELECT COUNT(*) as count FROM {table}")
                    count = count_df['count'].iloc[0] if not count_df.empty else 0
                    
                    status['database'][table] = {
                        'latest_timestamp': latest.isoformat() if latest else None,
                        'record_count': count
                    }
                except Exception as e:
                    status['database'][table] = {'error': str(e)}
            
            # Data summary
            try:
                summary = self.data_reader.get_data_summary()
                status['data_summary'] = summary
            except Exception as e:
                status['data_summary'] = {'error': str(e)}
            
            return status
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def export_training_data(self, output_path: str, feature_groups: list = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> bool:
        """
        Export training data
        
        Args:
            output_path: Path for output file
            feature_groups: List of feature groups to include
            start_time: Start of data range
            end_time: End of data range
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Exporting training data to {output_path}")
            
            if feature_groups is None:
                feature_groups = ['rpn', 'bin', 'dominance', 'signals']
            
            # Define feature columns
            feature_columns = []
            
            if 'rpn' in feature_groups:
                feature_columns.extend([
                    'risk_priority_number', 'lld_cwt_kf', 'fll_cwt_kf', 'fsl_cwt_kf',
                    'diff_ls_cwt_kf', 'is_rpn_extreme'
                ])
            
            if 'bin' in feature_groups:
                feature_columns.extend([
                    'bin_index', 'bin_momentum', 'bin_turn', 'bin_mm_direction'
                ])
            
            if 'dominance' in feature_groups:
                feature_columns.extend([
                    'dominance', 'dominance_duration_total', 'dominance_time', 'dominance_class'
                ])
            
            if 'signals' in feature_groups:
                feature_columns.extend([
                    'hit_ceiling_bottom', 'reverse_ceiling_bottom'
                ])
            
            # Get training data
            df = self.data_reader.get_training_data(
                feature_columns=feature_columns,
                target_column='signal_class',
                start_time=start_time,
                end_time=end_time,
                min_hours_between_signals=6
            )
            
            if df.empty:
                self.logger.warning("No training data available for export")
                return False
            
            # Export to file
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"✅ Exported {len(df)} training samples to {output_path}")
            
            # Log distribution
            if 'signal_class' in df.columns:
                dist = df['signal_class'].value_counts()
                self.logger.info(f"Signal distribution: {dist.to_dict()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
    
    def run_scheduled_updates(self, interval_seconds: int = 3600):
        """
        Run scheduled pipeline updates
        
        Args:
            interval_seconds: Interval between updates in seconds
        """
        self.logger.info(f"Starting scheduled updates every {interval_seconds} seconds")
        
        while True:
            try:
                self.logger.info("Running scheduled pipeline update...")
                
                # Run incremental pipeline (no full refresh)
                results = self.run_full_pipeline(full_refresh=False)
                
                self.logger.info(f"Scheduled update completed: {results}")
                
            except Exception as e:
                self.logger.error(f"Scheduled update failed: {e}")
            
            self.logger.info(f"Waiting {interval_seconds} seconds until next update...")
            time.sleep(interval_seconds)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BTC Feature Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--full-pipeline', action='store_true',
                             help='Run complete pipeline (download + preprocess + features)')
    action_group.add_argument('--download', action='store_true',
                             help='Run data download only')
    action_group.add_argument('--preprocess', action='store_true',
                             help='Run data preprocessing only')
    action_group.add_argument('--features', action='store_true',
                             help='Run feature engineering only')
    action_group.add_argument('--export', action='store_true',
                             help='Export training data')
    action_group.add_argument('--status', action='store_true',
                             help='Show pipeline status')
    action_group.add_argument('--schedule', action='store_true',
                             help='Run scheduled updates')
    action_group.add_argument('--init-db', action='store_true',
                             help='Initialize database schema')
    action_group.add_argument('--analyze', action='store_true',
                             help='Run statistical analysis on features')
    action_group.add_argument('--web-dashboard', action='store_true',
                             help='Start web dashboard for statistical analysis')
    
    # Options
    parser.add_argument('--config', type=str,
                       help='Configuration file path')
    parser.add_argument('--full-refresh', action='store_true',
                       help='Download all historical data (for --download)')
    parser.add_argument('--output', type=str, default='training_data.csv',
                       help='Output file path (for --export)')
    parser.add_argument('--feature-groups', nargs='+', 
                       choices=['rpn', 'bin', 'dominance', 'signals'],
                       default=['rpn', 'bin', 'dominance'],
                       help='Feature groups to include (for --export)')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Interval in seconds for scheduled updates (for --schedule)')
    parser.add_argument('--start-date', type=str,
                       help='Start date for processing (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for processing (YYYY-MM-DD)')
    parser.add_argument('--categories', nargs='+',
                       choices=['rpn', 'bin', 'dominance', 'signals', 'advanced'],
                       default=['rpn', 'bin', 'dominance', 'signals', 'advanced'],
                       help='Feature categories to analyze (for --analyze)')
    parser.add_argument('--export-format', choices=['json', 'csv'], default='json',
                       help='Export format for analysis results (for --analyze)')
    parser.add_argument('--port', type=int, default=8051,
                       help='Port for web dashboard (for --web-dashboard)')
    
    args = parser.parse_args()
    
    # Parse dates
    start_time = None
    end_time = None
    
    if args.start_date:
        try:
            start_time = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid start date format: {args.start_date}")
            return 1
    
    if args.end_date:
        try:
            end_time = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid end date format: {args.end_date}")
            return 1
    
    # Initialize pipeline
    try:
        pipeline = BTCFeaturePipeline(config_path=args.config)
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return 1
    
    # Execute requested action
    try:
        if args.init_db:
            success = pipeline.initialize_database()
            if success:
                print("✅ Database initialized successfully")
                return 0
            else:
                print("❌ Database initialization failed")
                return 1
        
        elif args.full_pipeline:
            results = pipeline.run_full_pipeline(full_refresh=args.full_refresh)
            print(json.dumps(results, indent=2, default=str))
            return 0
        
        elif args.download:
            results = pipeline.run_download(full_refresh=args.full_refresh)
            print(f"✅ Download completed: {results}")
            return 0
        
        elif args.preprocess:
            results = pipeline.run_preprocessing(start_time, end_time)
            print(f"✅ Preprocessing completed: {results}")
            return 0
        
        elif args.features:
            results = pipeline.run_feature_engineering(start_time, end_time)
            print(f"✅ Feature engineering completed: {results}")
            return 0
        
        elif args.export:
            success = pipeline.export_training_data(
                args.output, 
                args.feature_groups,
                start_time,
                end_time
            )
            if success:
                print(f"✅ Training data exported to {args.output}")
                return 0
            else:
                print("❌ Export failed")
                return 1
        
        elif args.status:
            status = pipeline.get_pipeline_status()
            print(json.dumps(status, indent=2, default=str))
            return 0
        
        elif args.schedule:
            pipeline.run_scheduled_updates(interval_seconds=args.interval)
            return 0
        
        elif args.analyze:
            # Import statistical analyzer
            from advanced_features.statistical_analyzer import run_statistical_analysis
            
            print("🔬 Running statistical analysis on features...")
            results = run_statistical_analysis(
                db_config=pipeline.config.get('database'),
                start_time=start_time,
                end_time=end_time,
                feature_categories=args.categories,
                export_format=args.export_format
            )
            
            # Print summary
            summary = results.get('executive_summary', {})
            print(f"✅ Statistical analysis completed:")
            print(f"   - Categories analyzed: {summary.get('total_categories', 0)}")
            print(f"   - Features analyzed: {summary.get('successful_analyses', 0)}")
            print(f"   - Success rate: {summary.get('successful_analyses', 0) / max(summary.get('total_features', 1), 1) * 100:.1f}%")
            
            return 0
        
        elif args.web_dashboard:
            # Import and start web interface
            from advanced_features.statistical_web_interface import create_statistical_web_interface
            
            print(f"🌐 Starting statistical analysis web dashboard on port {args.port}...")
            print(f"   Open your browser to: http://localhost:{args.port}")
            
            interface = create_statistical_web_interface(
                db_config=pipeline.config.get('database'),
                port=args.port
            )
            
            interface.run_server(debug=False, host='0.0.0.0')
            return 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())