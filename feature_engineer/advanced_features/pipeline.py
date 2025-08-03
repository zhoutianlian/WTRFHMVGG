"""
Feature Engineering Pipeline

This module provides the main pipeline orchestrator that uses the modular
advanced features system to process data in the correct execution order.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database_manager import DatabaseManager
from .registry import get_registry


class FeatureEngineeringPipeline:
    """
    Main feature engineering pipeline using modular advanced features system
    
    This orchestrates the execution of all features in the proper order,
    from existing features (RPN, binning, dominance, signals) to advanced features.
    """
    
    def __init__(self, db_manager: DatabaseManager, logger: Optional[logging.Logger] = None):
        self.db = db_manager
        self.logger = logger or logging.getLogger(__name__)
        self.registry = get_registry()
        
        # Log available features
        available_features = self.registry.list_features()
        self.logger.info(f"Initialized feature pipeline with {len(available_features)} features")
    
    def run_feature_engineering(self, start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              batch_size: int = 10000) -> Dict[str, int]:
        """
        Run complete feature engineering pipeline using modular system
        
        Args:
            start_time: Start of processing window
            end_time: End of processing window
            batch_size: Number of records to process at once
            
        Returns:
            Dictionary with counts of records processed per feature type
        """
        results = {}
        run_id = self.db.start_feature_run()
        
        try:
            # Determine processing window
            start_time, end_time = self._determine_processing_window(start_time, end_time)
            
            # Get processed data
            df = self.db.get_data_range('btc_processed_data', start_time, end_time)
            
            if df.empty:
                self.logger.warning("No processed data to engineer features from")
                return results
            
            self.logger.info(f"Engineering features for {len(df)} records from {df['time'].min()} to {df['time'].max()}")
            
            # Process in batches for memory efficiency
            total_records = 0
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size].copy()
                batch_num = i//batch_size + 1
                
                self.logger.info(f"Processing batch {batch_num} with {len(batch_df)} records")
                
                # Use modular pipeline
                batch_results = self._process_batch_with_modular_features(batch_df)
                
                # Aggregate results
                for feature_type, count in batch_results.items():
                    results[feature_type] = results.get(feature_type, 0) + count
                
                total_records += len(batch_df)
                self.logger.info(f"Completed batch {batch_num}")
            
            # Update run status
            self.db.end_feature_run(run_id, total_records, 'success')
            
            # Vacuum tables for performance
            self._vacuum_feature_tables()
            
            self.logger.info(f"Feature engineering complete: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            self.db.end_feature_run(run_id, 0, 'failed', str(e))
            raise
    
    def _determine_processing_window(self, start_time: Optional[datetime], 
                                   end_time: Optional[datetime]) -> tuple:
        """Determine the processing time window"""
        
        if start_time is None:
            # Check what's already been processed
            latest_rpn = self.db.get_latest_timestamp('btc_features_rpn')
            if latest_rpn:
                start_time = latest_rpn - timedelta(days=7)  # Reprocess last week for continuity
            else:
                # No features yet, start from earliest processed data
                earliest_processed = self.db.read_data("SELECT MIN(time) as min_time FROM btc_processed_data")
                if not earliest_processed.empty and earliest_processed['min_time'].iloc[0]:
                    start_time = earliest_processed['min_time'].iloc[0]
                else:
                    raise ValueError("No processed data available for feature engineering")
        
        return start_time, end_time
    
    def _process_batch_with_modular_features(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Process a batch using the modular features system
        
        This executes all features in the correct order (existing features first, then advanced)
        """
        results = {}
        
        # Get all available features
        available_features = self.registry.list_features()
        
        # Get optimal execution order (existing features first, then advanced)
        execution_order = self.registry.get_execution_order(available_features)
        
        # Validate dependencies
        missing_deps = self.registry.validate_feature_dependencies(
            available_features, df.columns.tolist()
        )
        
        if missing_deps:
            self.logger.warning(f"Some features have missing dependencies: {missing_deps}")
            # Filter out features with missing dependencies
            valid_features = [f for f in execution_order if f not in missing_deps]
        else:
            valid_features = execution_order
        
        self.logger.debug(f"Processing {len(valid_features)} features in order: {valid_features}")
        
        # Process features in optimal order
        current_df = df.copy()
        
        # Group features by category for database storage
        feature_categories = {
            'rpn': [],
            'bin': [], 
            'dominance': [],
            'signals': [],
            'advanced': []
        }
        
        # Execute all features in order
        for feature_name in valid_features:
            feature = self.registry.get_feature(feature_name)
            if feature is None:
                continue
            
            try:
                self.logger.debug(f"Processing feature: {feature_name}")
                
                # Apply feature calculation
                current_df = feature.calculate(current_df)
                
                # Categorize by feature type for storage
                if any(keyword in feature_name for keyword in ['cwt', 'kalman', 'ema', 'rpn', 'correlation']):
                    feature_categories['rpn'].append(feature_name)
                elif any(keyword in feature_name for keyword in ['kmeans', 'bin']):
                    feature_categories['bin'].append(feature_name)
                elif any(keyword in feature_name for keyword in ['dominance', 'regime']):
                    feature_categories['dominance'].append(feature_name)
                elif any(keyword in feature_name for keyword in ['beta', 'ceiling', 'reversal', 'signal']):
                    feature_categories['signals'].append(feature_name)
                else:
                    feature_categories['advanced'].append(feature_name)
                
            except Exception as e:
                self.logger.error(f"Error processing feature {feature_name}: {e}")
                continue
        
        # Store results by category
        results.update(self._store_features_by_category(current_df, feature_categories))
        
        return results
    
    def _store_features_by_category(self, df: pd.DataFrame, 
                                   feature_categories: Dict[str, List[str]]) -> Dict[str, int]:
        """Store features in appropriate database tables by category"""
        
        results = {}
        
        # RPN Features
        if feature_categories['rpn']:
            rpn_cols = ['time', 'price', 'risk_priority_number']
            rpn_feature_cols = [
                'lld_cwt_kf', 'fll_cwt_kf', 'fsl_cwt_kf', 'diff_ls_cwt_kf',
                'lld_cwt_kf_smooth', 'diff_ls_smooth', 'corr_case', 
                'delta_fll', 'delta_fsl', 'is_rpn_extreme'
            ]
            
            # Add available RPN feature columns
            for col in rpn_feature_cols:
                if col in df.columns:
                    rpn_cols.append(col)
            
            if len(rpn_cols) > 3:  # More than just time, price, risk_priority_number
                rpn_df = df[rpn_cols].copy()
                rpn_df = rpn_df.fillna(0)  # Fill NaN values
                
                self.db.insert_dataframe(rpn_df, 'btc_features_rpn', conflict_resolution='update')
                results['rpn'] = len(rpn_df)
                self.logger.debug(f"Stored {len(rpn_df)} RPN feature records")
        
        # Bin Features
        if feature_categories['bin'] and 'bin_index' in df.columns:
            bin_cols = ['time', 'bin_index', 'bin_momentum', 'bin_turn', 'bin_mm_direction']
            bin_cols = [col for col in bin_cols if col in df.columns]
            
            if len(bin_cols) > 1:
                bin_df = df[bin_cols].copy()
                bin_df = bin_df.fillna(0)
                
                self.db.insert_dataframe(bin_df, 'btc_features_bin', conflict_resolution='update')
                results['bin'] = len(bin_df)
                self.logger.debug(f"Stored {len(bin_df)} binning feature records")
        
        # Dominance Features
        if feature_categories['dominance'] and 'dominance' in df.columns:
            dom_cols = [
                'time', 'dominance', 'dominance_last', 'dominance_duration',
                'dominance_duration_total', 'dominance_time', 'dominance_class',
                'is_keep', 'is_strengthen'
            ]
            dom_cols = [col for col in dom_cols if col in df.columns]
            
            if len(dom_cols) > 1:
                dom_df = df[dom_cols].copy()
                dom_df = dom_df.fillna(0)
                
                # Fill categorical columns
                if 'dominance_class' in dom_df.columns:
                    dom_df['dominance_class'] = dom_df['dominance_class'].fillna('Congestion')
                
                self.db.insert_dataframe(dom_df, 'btc_features_dominance', conflict_resolution='update')
                results['dominance'] = len(dom_df)
                self.logger.debug(f"Stored {len(dom_df)} dominance feature records")
        
        # Signal Features
        if feature_categories['signals'] and 'signal_class' in df.columns:
            signal_cols = [
                'time', 'hit_ceiling_bottom', 'reverse_ceiling_bottom',
                'signal_l1', 'signal_l2', 'signal_l3', 'signal_class'
            ]
            signal_cols = [col for col in signal_cols if col in df.columns]
            
            if len(signal_cols) > 1:
                signal_df = df[signal_cols].copy()
                signal_df = signal_df.fillna(0)
                
                # Fill categorical columns
                if 'signal_class' in signal_df.columns:
                    signal_df['signal_class'] = signal_df['signal_class'].fillna('Neutral')
                
                self.db.insert_dataframe(signal_df, 'btc_features_signals', conflict_resolution='update')
                results['signals'] = len(signal_df)
                self.logger.debug(f"Stored {len(signal_df)} signal feature records")
        
        # Advanced Features
        if feature_categories['advanced']:
            # Get all advanced feature columns
            advanced_cols = ['time']
            for col in df.columns:
                if col not in ['time'] and any(keyword in col.lower() for keyword in [
                    'kama', 'spike', 'roc', 'velocity', 'acceleration', 'rsi', 'stoch', 
                    'macd', 'kalman_slope', 'skewness', 'kurtosis', 'jarque_bera', 
                    'vol_adaptive', 'vol_garch', 'cv', 'fat_tails', 'non_normal'
                ]):
                    advanced_cols.append(col)
            
            if len(advanced_cols) > 1:
                adv_df = df[advanced_cols].copy()
                adv_df = adv_df.fillna(0)
                
                self.db.insert_dataframe(adv_df, 'btc_features_advanced', conflict_resolution='update')
                results['advanced'] = len(adv_df)
                self.logger.debug(f"Stored {len(adv_df)} advanced feature records")
        
        return results
    
    def _vacuum_feature_tables(self):
        """Vacuum feature tables for performance"""
        feature_tables = [
            'btc_features_rpn', 'btc_features_bin', 'btc_features_dominance', 
            'btc_features_signals', 'btc_features_advanced'
        ]
        
        for table in feature_tables:
            try:
                self.db.vacuum_analyze(table)
                self.logger.debug(f"Vacuumed table: {table}")
            except Exception as e:
                self.logger.warning(f"Failed to vacuum table {table}: {e}")


# Create the FeatureEngineer alias for backward compatibility
FeatureEngineer = FeatureEngineeringPipeline


# Convenience function for command-line usage
def run_feature_engineering(db_config: Optional[dict] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None):
    """Run feature engineering pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_config)
        pipeline = FeatureEngineeringPipeline(db, logger)
        
        results = pipeline.run_feature_engineering(start_time, end_time)
        logger.info(f"Feature engineering complete: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    run_feature_engineering()