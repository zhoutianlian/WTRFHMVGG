import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database_manager import DatabaseManager


class BTCDataReader:
    """Convenient interface for reading BTC data and features from PostgreSQL database"""
    
    def __init__(self, db_manager: DatabaseManager, logger: Optional[logging.Logger] = None):
        self.db = db_manager
        self.logger = logger or logging.getLogger(__name__)
    
    def get_raw_data(self, start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get raw price and liquidation data"""
        query = """
            SELECT 
                p.time,
                p.price,
                l.long_liquidations_dominance,
                l.futures_long_liquidations,
                l.futures_short_liquidations
            FROM btc_price_data p
            LEFT JOIN btc_liquidations l ON p.time = l.time
            WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND p.time >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND p.time <= %s"
            params.append(end_time)
        
        query += " ORDER BY p.time"
        
        return self.db.read_data(query, tuple(params) if params else None)
    
    def get_processed_data(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get preprocessed data"""
        return self.db.get_data_range('btc_processed_data', start_time, end_time)
    
    def get_all_features(self, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get all features joined together"""
        query = """
            SELECT 
                p.time,
                p.price,
                p.lld_normal,
                p.fll_normal,
                p.fsl_normal,
                p.diff_ls_normal,
                r.risk_priority_number,
                r.lld_cwt_kf,
                r.fll_cwt_kf,
                r.fsl_cwt_kf,
                r.diff_ls_cwt_kf,
                r.lld_cwt_kf_smooth,
                r.diff_ls_smooth,
                r.corr_case,
                r.delta_fll,
                r.delta_fsl,
                r.is_rpn_extreme,
                b.bin_index,
                b.bin_momentum,
                b.bin_turn,
                b.bin_mm_direction,
                d.dominance,
                d.dominance_last,
                d.dominance_duration,
                d.dominance_duration_total,
                d.dominance_time,
                d.dominance_class,
                d.is_keep,
                d.is_strengthen,
                s.hit_ceiling_bottom,
                s.reverse_ceiling_bottom,
                s.signal_l1,
                s.signal_l2,
                s.signal_l3,
                s.signal_class,
                a.fll_kama,
                a.fsl_kama,
                a.price_kama,
                a.fll_spike_kama,
                a.fsl_spike_kama,
                a.price_spike_ema,
                a.fll_velocity,
                a.fll_acceleration,
                a.price_velocity,
                a.price_acceleration,
                a.price_rsi,
                a.price_stoch_k,
                a.price_stoch_d,
                a.price_macd,
                a.price_macd_signal,
                a.price_macd_histogram,
                a.kalman_slope,
                a.kalman_slope_filtered_price,
                a.kalman_slope_innovation,
                a.kalman_slope_smooth,
                a.kalman_slope_direction,
                a.kalman_slope_acceleration,
                a.kalman_slope_volatility,
                a.price_skewness,
                a.price_skew_signal,
                a.price_kurtosis,
                a.price_excess_kurtosis,
                a.price_fat_tails,
                a.price_jarque_bera,
                a.price_non_normal,
                a.price_cv,
                a.vol_adaptive,
                a.vol_garch
            FROM btc_processed_data p
            LEFT JOIN btc_features_rpn r ON p.time = r.time
            LEFT JOIN btc_features_bin b ON p.time = b.time
            LEFT JOIN btc_features_dominance d ON p.time = d.time
            LEFT JOIN btc_features_signals s ON p.time = s.time
            LEFT JOIN btc_features_advanced a ON p.time = a.time
            WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND p.time >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND p.time <= %s"
            params.append(end_time)
        
        query += " ORDER BY p.time"
        
        return self.db.read_data(query, tuple(params) if params else None)
    
    def get_latest_signals(self, hours: int = 24) -> pd.DataFrame:
        """Get latest signals from the past N hours"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT 
                s.time,
                p.price,
                r.risk_priority_number,
                b.bin_index,
                d.dominance_class,
                s.signal_class,
                s.hit_ceiling_bottom,
                s.reverse_ceiling_bottom,
                s.signal_l1,
                s.signal_l2,
                s.signal_l3
            FROM btc_features_signals s
            JOIN btc_processed_data p ON s.time = p.time
            LEFT JOIN btc_features_rpn r ON s.time = r.time
            LEFT JOIN btc_features_bin b ON s.time = b.time
            LEFT JOIN btc_features_dominance d ON s.time = d.time
            WHERE s.time >= %s
                AND (s.signal_l1 != 0 OR s.signal_l2 != 0 OR s.signal_l3 != 0)
            ORDER BY s.time DESC
        """
        
        return self.db.read_data(query, (start_time,))
    
    def get_feature_stats(self, feature_table: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistics for a feature table"""
        try:
            # Get date range and count
            date_query = f"SELECT MIN(time) as min_time, MAX(time) as max_time, COUNT(*) as count FROM {feature_table}"
            params = []
            where_clause = " WHERE 1=1"
            
            if start_time:
                where_clause += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                where_clause += " AND time <= %s"
                params.append(end_time)
            
            date_query += where_clause
            
            df_stats = self.db.read_data(date_query, tuple(params) if params else None)
            
            if df_stats.empty:
                return {'error': 'No data found'}
            
            stats = {
                'table': feature_table,
                'min_time': df_stats['min_time'].iloc[0],
                'max_time': df_stats['max_time'].iloc[0],
                'record_count': df_stats['count'].iloc[0]
            }
            
            # Get column info
            col_query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """
            df_cols = self.db.read_data(col_query, (feature_table,))
            stats['columns'] = df_cols.to_dict('records') if not df_cols.empty else []
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting feature stats for {feature_table}: {e}")
            return {'error': str(e)}
    
    def get_training_data(self, feature_columns: List[str],
                         target_column: str = 'signal_class',
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         min_hours_between_signals: int = 6,
                         balance_classes: bool = False) -> pd.DataFrame:
        """
        Get data formatted for training
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            start_time: Start of data range
            end_time: End of data range
            min_hours_between_signals: Minimum hours between signal samples
            balance_classes: Whether to balance signal classes
            
        Returns:
            DataFrame with features and target
        """
        try:
            # Get all features
            df = self.get_all_features(start_time, end_time)
            
            if df.empty:
                self.logger.warning("No features found for training data")
                return pd.DataFrame()
            
            # Filter to only rows with signals if target is signal_class
            if target_column == 'signal_class':
                df = df[df['signal_class'] != 'Neutral'].copy()
            
            # Filter by minimum time between signals
            if min_hours_between_signals > 0:
                df = self._filter_by_time_spacing(df, min_hours_between_signals)
            
            # Select and validate columns
            all_columns = ['time'] + feature_columns + [target_column]
            available_columns = [col for col in all_columns if col in df.columns]
            
            if target_column not in available_columns:
                self.logger.error(f"Target column {target_column} not found")
                return pd.DataFrame()
            
            missing_features = [col for col in feature_columns if col not in available_columns]
            if missing_features:
                self.logger.warning(f"Missing feature columns: {missing_features}")
            
            result_df = df[available_columns].copy()
            
            # Handle missing values
            result_df = self._handle_training_data_missing_values(result_df, feature_columns)
            
            # Balance classes if requested
            if balance_classes and target_column == 'signal_class':
                result_df = self._balance_signal_classes(result_df, target_column)
            
            self.logger.info(f"Prepared training data: {len(result_df)} samples with {len(feature_columns)} features")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def _filter_by_time_spacing(self, df: pd.DataFrame, min_hours: int) -> pd.DataFrame:
        """Filter dataframe to maintain minimum time spacing between samples"""
        if df.empty:
            return df
        
        df = df.sort_values('time').copy()
        mask = [True]  # Always include first row
        last_time = df['time'].iloc[0]
        
        for i in range(1, len(df)):
            current_time = df['time'].iloc[i]
            if (current_time - last_time).total_seconds() >= min_hours * 3600:
                mask.append(True)
                last_time = current_time
            else:
                mask.append(False)
        
        filtered_df = df[mask].copy()
        self.logger.info(f"Time filtering: {len(df)} -> {len(filtered_df)} samples (min {min_hours}h spacing)")
        
        return filtered_df
    
    def _handle_training_data_missing_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Handle missing values in training data"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Fill missing values for numeric features
        numeric_features = [col for col in feature_columns if col in df.columns and df[col].dtype in ['float64', 'int64']]
        
        for col in numeric_features:
            if df[col].isnull().any():
                # Forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # Fill any remaining with median
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        # Fill categorical features
        categorical_features = [col for col in feature_columns if col in df.columns and df[col].dtype == 'object']
        
        for col in categorical_features:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df
    
    def _balance_signal_classes(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Balance signal classes for training"""
        if df.empty or target_column not in df.columns:
            return df
        
        class_counts = df[target_column].value_counts()
        min_count = class_counts.min()
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = df[df[target_column] == class_name]
            sampled_df = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        self.logger.info(f"Class balancing: {len(df)} -> {len(balanced_df)} samples")
        
        return balanced_df
    
    def export_to_csv(self, output_dir: str, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None):
        """Export all data to CSV files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            export_tasks = [
                ('raw_data', lambda: self.get_raw_data(start_time, end_time)),
                ('processed_data', lambda: self.get_processed_data(start_time, end_time)),
                ('all_features', lambda: self.get_all_features(start_time, end_time)),
                ('latest_signals', lambda: self.get_latest_signals(hours=24*7))
            ]
            
            for name, getter in export_tasks:
                self.logger.info(f"Exporting {name}...")
                try:
                    df = getter()
                    
                    if not df.empty:
                        filepath = os.path.join(output_dir, f"{name}.csv")
                        df.to_csv(filepath, index=False)
                        self.logger.info(f"Exported {len(df)} rows to {filepath}")
                    else:
                        self.logger.warning(f"No data found for {name}")
                        
                except Exception as e:
                    self.logger.error(f"Error exporting {name}: {e}")
            
            self.logger.info(f"Export completed to directory: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in export process: {e}")
    
    def get_performance_metrics(self, hours: int = 24*7) -> pd.DataFrame:
        """Get performance metrics for signals"""
        query = """
            WITH signal_returns AS (
                SELECT 
                    s.time,
                    s.signal_class,
                    p1.price as signal_price,
                    p2.price as price_1h,
                    p3.price as price_6h,
                    p4.price as price_24h,
                    CASE 
                        WHEN p2.price IS NOT NULL THEN 
                            (p2.price - p1.price) / p1.price * 100
                        ELSE NULL 
                    END as return_1h,
                    CASE 
                        WHEN p3.price IS NOT NULL THEN 
                            (p3.price - p1.price) / p1.price * 100
                        ELSE NULL 
                    END as return_6h,
                    CASE 
                        WHEN p4.price IS NOT NULL THEN 
                            (p4.price - p1.price) / p1.price * 100
                        ELSE NULL 
                    END as return_24h
                FROM btc_features_signals s
                JOIN btc_processed_data p1 ON s.time = p1.time
                LEFT JOIN btc_processed_data p2 ON p2.time = s.time + INTERVAL '1 hour'
                LEFT JOIN btc_processed_data p3 ON p3.time = s.time + INTERVAL '6 hours'
                LEFT JOIN btc_processed_data p4 ON p4.time = s.time + INTERVAL '24 hours'
                WHERE s.time >= NOW() - INTERVAL '%s hours'
                    AND s.signal_class != 'Neutral'
            )
            SELECT 
                signal_class,
                COUNT(*) as signal_count,
                AVG(return_1h) as avg_return_1h,
                AVG(return_6h) as avg_return_6h,
                AVG(return_24h) as avg_return_24h,
                STDDEV(return_1h) as stddev_return_1h,
                STDDEV(return_6h) as stddev_return_6h,
                STDDEV(return_24h) as stddev_return_24h,
                MIN(return_1h) as min_return_1h,
                MAX(return_1h) as max_return_1h,
                COALESCE(
                    SUM(CASE WHEN return_1h > 0 THEN 1 ELSE 0 END)::FLOAT / 
                    NULLIF(COUNT(CASE WHEN return_1h IS NOT NULL THEN 1 END), 0), 
                    0
                ) as win_rate_1h,
                COALESCE(
                    SUM(CASE WHEN return_6h > 0 THEN 1 ELSE 0 END)::FLOAT / 
                    NULLIF(COUNT(CASE WHEN return_6h IS NOT NULL THEN 1 END), 0), 
                    0
                ) as win_rate_6h,
                COALESCE(
                    SUM(CASE WHEN return_24h > 0 THEN 1 ELSE 0 END)::FLOAT / 
                    NULLIF(COUNT(CASE WHEN return_24h IS NOT NULL THEN 1 END), 0), 
                    0
                ) as win_rate_24h
            FROM signal_returns
            GROUP BY signal_class
            ORDER BY signal_class
        """
        
        try:
            return self.db.read_data(query, (hours,))
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return pd.DataFrame()
    
    def get_combined_features(self, start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get combined data from all feature tables for dashboard visualization
        
        This method merges data from processed data, RPN, binning, dominance, and signal features
        to provide a comprehensive dataset for visualization.
        """
        try:
            # Start with processed data (includes price and basic liquidation data)
            base_query = """
                SELECT 
                    time, price, 
                    lld_normal, fll_normal, fsl_normal, diff_ls_normal
                FROM btc_processed_data
                WHERE 1=1
            """
            params = []
            
            if start_time:
                base_query += " AND time >= %s"
                params.append(start_time)
            
            if end_time:
                base_query += " AND time <= %s"
                params.append(end_time)
            
            base_query += " ORDER BY time"
            
            df = self.db.read_data(base_query, tuple(params) if params else None)
            
            if df.empty:
                self.logger.warning("No base processed data found")
                return pd.DataFrame()
            
            # Add RPN features
            try:
                rpn_query = """
                    SELECT 
                        time, risk_priority_number, lld_cwt_kf, fll_cwt_kf, fsl_cwt_kf,
                        diff_ls_cwt_kf, corr_case, delta_fll, delta_fsl, is_rpn_extreme
                    FROM btc_features_rpn
                    WHERE time >= %s AND time <= %s
                    ORDER BY time
                """
                rpn_df = self.db.read_data(rpn_query, (df['time'].min(), df['time'].max()))
                
                if not rpn_df.empty:
                    df = pd.merge(df, rpn_df, on='time', how='left')
                    self.logger.debug(f"Merged {len(rpn_df)} RPN feature records")
                
            except Exception as e:
                self.logger.warning(f"Could not load RPN features: {e}")
            
            # Add binning features
            try:
                bin_query = """
                    SELECT 
                        time, bin_index, bin_momentum, bin_turn, bin_mm_direction
                    FROM btc_features_bin
                    WHERE time >= %s AND time <= %s
                    ORDER BY time
                """
                bin_df = self.db.read_data(bin_query, (df['time'].min(), df['time'].max()))
                
                if not bin_df.empty:
                    df = pd.merge(df, bin_df, on='time', how='left')
                    self.logger.debug(f"Merged {len(bin_df)} binning feature records")
                
            except Exception as e:
                self.logger.warning(f"Could not load binning features: {e}")
            
            # Add dominance features
            try:
                dom_query = """
                    SELECT 
                        time, dominance, dominance_last, dominance_duration,
                        dominance_duration_total, dominance_time, dominance_class,
                        is_keep, is_strengthen
                    FROM btc_features_dominance
                    WHERE time >= %s AND time <= %s
                    ORDER BY time
                """
                dom_df = self.db.read_data(dom_query, (df['time'].min(), df['time'].max()))
                
                if not dom_df.empty:
                    df = pd.merge(df, dom_df, on='time', how='left')
                    self.logger.debug(f"Merged {len(dom_df)} dominance feature records")
                
            except Exception as e:
                self.logger.warning(f"Could not load dominance features: {e}")
            
            # Add signal features
            try:
                signal_query = """
                    SELECT 
                        time, hit_ceiling_bottom, reverse_ceiling_bottom,
                        signal_l1, signal_l2, signal_l3, signal_class
                    FROM btc_features_signals
                    WHERE time >= %s AND time <= %s
                    ORDER BY time
                """
                signal_df = self.db.read_data(signal_query, (df['time'].min(), df['time'].max()))
                
                if not signal_df.empty:
                    df = pd.merge(df, signal_df, on='time', how='left')
                    self.logger.debug(f"Merged {len(signal_df)} signal feature records")
                
            except Exception as e:
                self.logger.warning(f"Could not load signal features: {e}")
            
            # Add advanced features if available
            try:
                adv_query = """
                    SELECT *
                    FROM btc_features_advanced
                    WHERE time >= %s AND time <= %s
                    ORDER BY time
                """
                adv_df = self.db.read_data(adv_query, (df['time'].min(), df['time'].max()))
                
                if not adv_df.empty:
                    # Remove duplicate time column
                    adv_df = adv_df.drop(columns=['time'])
                    # Merge by index (assumes same time alignment)
                    df = pd.concat([df, adv_df], axis=1)
                    self.logger.debug(f"Merged {len(adv_df)} advanced feature records")
                
            except Exception as e:
                self.logger.debug(f"Advanced features not available: {e}")
            
            # Clean and prepare data
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            # Fill missing values for visualization
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Fill categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != 'time':
                    df[col] = df[col].fillna('None')
            
            self.logger.info(f"Combined features: {len(df)} records with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting combined features: {e}")
            return pd.DataFrame()

    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        try:
            summary = {}
            
            # Table counts and date ranges
            tables = ['btc_price_data', 'btc_liquidations', 'btc_processed_data', 
                     'btc_features_rpn', 'btc_features_bin', 'btc_features_dominance', 'btc_features_signals']
            
            for table in tables:
                try:
                    stats = self.get_feature_stats(table)
                    summary[table] = stats
                except Exception as e:
                    summary[table] = {'error': str(e)}
            
            # Signal distribution
            try:
                signal_dist = self.db.read_data("""
                    SELECT signal_class, COUNT(*) as count 
                    FROM btc_features_signals 
                    GROUP BY signal_class 
                    ORDER BY count DESC
                """)
                summary['signal_distribution'] = signal_dist.to_dict('records') if not signal_dist.empty else []
            except Exception as e:
                summary['signal_distribution'] = {'error': str(e)}
            
            # Recent performance
            try:
                perf = self.get_performance_metrics(hours=24*30)  # Last 30 days
                summary['recent_performance'] = perf.to_dict('records') if not perf.empty else []
            except Exception as e:
                summary['recent_performance'] = {'error': str(e)}
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {'error': str(e)}


# Convenience functions for command-line usage
def read_latest_features(db_config: Optional[dict] = None, hours: int = 24):
    """Read latest features from database"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_config)
        reader = BTCDataReader(db, logger)
        
        # Get latest features
        df = reader.get_all_features(
            start_time=datetime.now() - timedelta(hours=hours)
        )
        
        logger.info(f"Retrieved {len(df)} feature records from last {hours} hours")
        
        # Get latest signals
        signals_df = reader.get_latest_signals(hours)
        logger.info(f"Found {len(signals_df)} signals in the last {hours} hours")
        
        if not signals_df.empty:
            logger.info("\nLatest signals:")
            for _, row in signals_df.head(10).iterrows():
                logger.info(f"{row['time']}: {row['signal_class']} (Price: ${row['price']:.2f})")
        
        return df, signals_df
        
    except Exception as e:
        logger.error(f"Error reading latest features: {e}")
        return pd.DataFrame(), pd.DataFrame()


def export_training_data(db_config: Optional[dict] = None, 
                        output_path: str = "training_data.csv",
                        feature_groups: List[str] = None):
    """Export data for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if feature_groups is None:
        feature_groups = ['rpn', 'bin', 'dominance']
    
    try:
        db = DatabaseManager(db_config)
        reader = BTCDataReader(db, logger)
        
        # Define feature columns based on groups
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
        df = reader.get_training_data(
            feature_columns=feature_columns,
            target_column='signal_class',
            min_hours_between_signals=6,
            balance_classes=False
        )
        
        if not df.empty:
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} training samples to {output_path}")
            
            # Show distribution
            if 'signal_class' in df.columns:
                dist = df['signal_class'].value_counts()
                logger.info(f"Signal distribution: {dist.to_dict()}")
        else:
            logger.warning("No training data found")
        
        return df
        
    except Exception as e:
        logger.error(f"Error exporting training data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'export':
            export_training_data()
        elif sys.argv[1] == 'latest':
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            read_latest_features(hours=hours)
        elif sys.argv[1] == 'summary':
            logging.basicConfig(level=logging.INFO)
            db = DatabaseManager()
            reader = BTCDataReader(db)
            summary = reader.get_data_summary()
            print("Data Summary:", summary)
    else:
        # Default: read latest 24 hours
        read_latest_features(hours=24)