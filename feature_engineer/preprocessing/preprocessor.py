import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database_manager import DatabaseManager


class DataPreprocessor:
    """Data preprocessing for BTC feature engineering pipeline"""
    
    def __init__(self, db_manager: DatabaseManager, logger: Optional[logging.Logger] = None):
        self.db = db_manager
        self.logger = logger or logging.getLogger(__name__)
    
    def run_preprocessing(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         batch_size: int = 10000) -> Dict[str, int]:
        """
        Run complete data preprocessing pipeline
        
        Args:
            start_time: Start of processing window
            end_time: End of processing window
            batch_size: Number of records to process at once
            
        Returns:
            Dictionary with counts of records processed
        """
        results = {}
        
        try:
            # Determine time range for preprocessing
            if start_time is None:
                # Check what's already been processed
                latest_processed = self.db.get_latest_timestamp('btc_processed_data')
                if latest_processed:
                    start_time = latest_processed - timedelta(hours=24)  # Reprocess last day for continuity
                else:
                    # No processed data, start from earliest raw data
                    earliest_price = self.db.read_data("SELECT MIN(time) as min_time FROM btc_price_data")
                    if not earliest_price.empty and earliest_price['min_time'].iloc[0]:
                        start_time = earliest_price['min_time'].iloc[0]
                    else:
                        self.logger.error("No raw data available for preprocessing")
                        return results
            
            if end_time is None:
                end_time = datetime.now()
            
            self.logger.info(f"Preprocessing data from {start_time} to {end_time}")
            
            # Get raw data
            raw_data = self._get_combined_raw_data(start_time, end_time)
            
            if raw_data.empty:
                self.logger.warning("No raw data found for preprocessing")
                return results
            
            self.logger.info(f"Processing {len(raw_data)} raw records")
            
            # Process in batches
            total_processed = 0
            for i in range(0, len(raw_data), batch_size):
                batch_df = raw_data.iloc[i:i+batch_size].copy()
                
                # Apply preprocessing steps
                processed_batch = self._preprocess_batch(batch_df)
                
                if not processed_batch.empty:
                    # Store processed data
                    self.db.insert_dataframe(processed_batch, 'btc_processed_data', conflict_resolution='update')
                    total_processed += len(processed_batch)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}: {len(processed_batch)} records")
            
            results['processed_records'] = total_processed
            
            # Vacuum table for performance
            self.db.vacuum_analyze('btc_processed_data')
            
            self.logger.info(f"Preprocessing complete: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise
    
    def _get_combined_raw_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Combine price and liquidation data for preprocessing
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Combined DataFrame with all raw data
        """
        try:
            # Get price data
            price_df = self.db.get_data_range('btc_price_data', start_time, end_time, ['time', 'price'])
            
            if price_df.empty:
                self.logger.warning("No price data found in specified range")
                return pd.DataFrame()
            
            # Get liquidation data
            liquidation_df = self.db.get_data_range('btc_liquidations', start_time, end_time)
            
            if liquidation_df.empty:
                self.logger.warning("No liquidation data found in specified range")
                # Return just price data if liquidations not available
                return price_df
            
            # Merge on time
            combined_df = pd.merge(price_df, liquidation_df, on='time', how='outer', suffixes=('', '_liq'))
            
            # Sort by time
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            self.logger.info(f"Combined {len(price_df)} price records with {len(liquidation_df)} liquidation records")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error combining raw data: {e}")
            return pd.DataFrame()
    
    def _preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to a batch of data
        
        Args:
            df: Raw data batch
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            # 1. Handle missing values
            df = self._handle_missing_values(df)
            
            # 2. Normalize liquidation data
            df = self._normalize_liquidation_data(df)
            
            # 3. Smooth outliers
            df = self._smooth_outliers(df)
            
            # 4. Calculate derived metrics
            df = self._calculate_derived_metrics(df)
            
            # 5. Select final columns for processed data
            output_columns = [
                'time', 'price', 'lld_normal', 'fll_normal', 
                'fsl_normal', 'diff_ls_normal'
            ]
            
            # Only keep columns that exist
            existing_columns = [col for col in output_columns if col in df.columns]
            df = df[existing_columns]
            
            # Remove any remaining NaN rows
            df = df.dropna(subset=['time', 'price'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in batch preprocessing: {e}")
            return pd.DataFrame()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        df = df.copy()
        
        # Forward fill then backward fill for liquidation data
        liquidation_cols = [
            'long_liquidations_dominance', 
            'futures_long_liquidations', 
            'futures_short_liquidations'
        ]
        
        for col in liquidation_cols:
            if col in df.columns:
                # Count initial NaNs
                initial_nans = df[col].isnull().sum()
                
                # Apply interpolation for small gaps
                df[col] = df[col].interpolate(method='linear', limit=3)
                
                # Forward fill remaining gaps
                df[col] = df[col].fillna(method='ffill')
                
                # Backward fill any remaining
                df[col] = df[col].fillna(method='bfill')
                
                # Fill any still remaining with median
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                
                final_nans = df[col].isnull().sum()
                if initial_nans > 0:
                    self.logger.debug(f"Filled {initial_nans - final_nans} NaN values in {col}")
        
        return df
    
    def _normalize_liquidation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize liquidation data as in original preprocessing"""
        df = df.copy()
        
        try:
            # Create normalized columns
            if 'long_liquidations_dominance' in df.columns:
                df['lld_normal'] = df['long_liquidations_dominance']
            
            if 'futures_long_liquidations' in df.columns:
                df['fll_normal'] = df['futures_long_liquidations']
            
            if 'futures_short_liquidations' in df.columns:
                df['fsl_normal'] = df['futures_short_liquidations']
            
            # Apply log transformation to reduce skewness for large values
            for col in ['fll_normal', 'fsl_normal']:
                if col in df.columns:
                    # Add small constant to avoid log(0)
                    df[col] = np.log1p(df[col].clip(lower=0))
            
            # Normalize lld_normal to 0-1 range if needed
            if 'lld_normal' in df.columns:
                df['lld_normal'] = df['lld_normal'].clip(0, 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in liquidation normalization: {e}")
            return df
    
    def _smooth_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth outliers using rolling statistics"""
        df = df.copy()
        
        # Define columns to smooth
        smooth_cols = ['fll_normal', 'fsl_normal', 'lld_normal']
        
        for col in smooth_cols:
            if col in df.columns:
                try:
                    # Calculate rolling statistics
                    rolling_median = df[col].rolling(window=24, center=True, min_periods=1).median()
                    rolling_std = df[col].rolling(window=24, center=True, min_periods=1).std()
                    
                    # Define outlier threshold (3 standard deviations)
                    threshold = 3
                    lower_bound = rolling_median - threshold * rolling_std
                    upper_bound = rolling_median + threshold * rolling_std
                    
                    # Identify outliers
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        # Replace outliers with rolling median
                        df.loc[outliers, col] = rolling_median[outliers]
                        self.logger.debug(f"Smoothed {outlier_count} outliers in {col}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not smooth outliers in {col}: {e}")
        
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics"""
        df = df.copy()
        
        try:
            # Calculate difference between long and short liquidations
            if 'fll_normal' in df.columns and 'fsl_normal' in df.columns:
                df['diff_ls_normal'] = df['fll_normal'] - df['fsl_normal']
            
            # Calculate total liquidations
            if 'fll_normal' in df.columns and 'fsl_normal' in df.columns:
                df['total_liquidations'] = df['fll_normal'] + df['fsl_normal']
            
            # Calculate liquidation ratio (alternative to dominance)
            if 'fll_normal' in df.columns and 'total_liquidations' in df.columns:
                df['liquidation_ratio'] = df['fll_normal'] / (df['total_liquidations'] + 1e-10)
            
            # If lld_normal is missing, calculate from ratio
            if 'lld_normal' not in df.columns and 'liquidation_ratio' in df.columns:
                df['lld_normal'] = df['liquidation_ratio']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {e}")
            return df
    
    def validate_processed_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Validate processed data quality
        
        Args:
            start_time: Start of validation window
            end_time: End of validation window
            
        Returns:
            Validation results
        """
        results = {
            'validation_passed': False,
            'issues': [],
            'statistics': {}
        }
        
        try:
            # Get processed data
            df = self.db.get_data_range('btc_processed_data', start_time, end_time)
            
            if df.empty:
                results['issues'].append("No processed data found")
                return results
            
            # Check for required columns
            required_cols = ['time', 'price', 'lld_normal']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                results['issues'].append(f"Missing required columns: {missing_cols}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            critical_nulls = null_counts[null_counts > 0]
            if not critical_nulls.empty:
                results['issues'].append(f"Null values found: {critical_nulls.to_dict()}")
            
            # Check data ranges
            for col in ['lld_normal']:
                if col in df.columns:
                    col_min, col_max = df[col].min(), df[col].max()
                    if col == 'lld_normal' and (col_min < 0 or col_max > 1):
                        results['issues'].append(f"{col} values outside expected range [0,1]: [{col_min:.4f}, {col_max:.4f}]")
            
            # Check temporal continuity
            df_sorted = df.sort_values('time')
            time_diffs = df_sorted['time'].diff().dt.total_seconds() / 3600  # Convert to hours
            large_gaps = time_diffs[time_diffs > 2]  # Gaps larger than 2 hours
            if not large_gaps.empty:
                results['issues'].append(f"Found {len(large_gaps)} time gaps larger than 2 hours")
            
            # Calculate statistics
            results['statistics'] = {
                'record_count': len(df),
                'date_range': {
                    'start': df['time'].min().isoformat() if not df.empty else None,
                    'end': df['time'].max().isoformat() if not df.empty else None
                },
                'column_stats': {}
            }
            
            # Column-wise statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'id':  # Skip ID column
                    results['statistics']['column_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'null_count': int(df[col].isnull().sum())
                    }
            
            # Overall validation result
            results['validation_passed'] = len(results['issues']) == 0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            results['issues'].append(f"Validation error: {str(e)}")
            return results
    
    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        try:
            # Get table info
            processed_info = self.db.get_table_info('btc_processed_data')
            
            # Get data quality metrics
            quality_query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(time) as earliest_record,
                    MAX(time) as latest_record,
                    AVG(CASE WHEN lld_normal IS NULL THEN 1 ELSE 0 END) as lld_null_rate,
                    AVG(CASE WHEN fll_normal IS NULL THEN 1 ELSE 0 END) as fll_null_rate,
                    AVG(CASE WHEN fsl_normal IS NULL THEN 1 ELSE 0 END) as fsl_null_rate
                FROM btc_processed_data
            """
            
            quality_df = self.db.read_data(quality_query)
            
            return {
                'table_info': processed_info,
                'data_quality': quality_df.to_dict('records')[0] if not quality_df.empty else {},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting preprocessing statistics: {e}")
            return {'error': str(e)}


# Convenience function for command-line usage
def run_preprocessing(db_config: Optional[Dict] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None):
    """Run data preprocessing pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_config)
        preprocessor = DataPreprocessor(db, logger)
        
        results = preprocessor.run_preprocessing(start_time, end_time)
        logger.info(f"Preprocessing complete: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    run_preprocessing()