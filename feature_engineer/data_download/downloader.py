import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'feature_engineering_orig', 'download'))

from feature_engineering_orig.download.download_glassnode import GlassnodeClient, get_n_days_ago
from db.database_manager import DatabaseManager


class BTCDataDownloader:
    """Download BTC data from Glassnode and store in PostgreSQL"""
    
    def __init__(self, db_manager: DatabaseManager, logger: Optional[logging.Logger] = None):
        self.db = db_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize Glassnode client
        try:
            self.client = GlassnodeClient(self.logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize GlassnodeClient: {e}")
            raise
    
    def download_metric(self, endpoint: str, metric_name: str, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download a single metric from Glassnode
        
        Args:
            endpoint: Glassnode API endpoint
            metric_name: Name for the metric column
            start_time: Start timestamp (uses last available if None)
            end_time: End timestamp (uses current time if None)
            
        Returns:
            DataFrame with time and metric columns
        """
        try:
            # Reset client parameters
            self.client.reset_parameters()
            
            # Set parameters
            self.client.set_parameter('a', 'BTC')
            self.client.set_parameter('i', '1h')
            
            # Set time range
            if end_time:
                self.client.set_parameter('u', int(end_time.timestamp()))
            else:
                self.client.set_parameter('u', get_n_days_ago(0))
            
            if start_time:
                self.client.set_parameter('s', int(start_time.timestamp()))
            
            # Set endpoint and download
            self.client.endpoint = endpoint
            df = self.client.get_dataframe()
            
            if df is None or df.empty:
                self.logger.warning(f"Empty data for {metric_name} from {endpoint}")
                return pd.DataFrame()
            
            # Rename columns for consistency
            df.columns = ['time', metric_name]
            
            # Ensure proper data types
            df['time'] = pd.to_datetime(df['time'])
            df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
            
            self.logger.info(f"Downloaded {len(df)} records for {metric_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading {metric_name} from {endpoint}: {e}")
            return pd.DataFrame()
    
    def get_download_start_time(self, table_name: str, default_days_back: int = 30) -> datetime:
        """
        Determine start time for download based on existing data
        
        Args:
            table_name: Table to check for existing data
            default_days_back: Default days to go back if no data exists
            
        Returns:
            Start timestamp for download
        """
        try:
            latest_time = self.db.get_latest_timestamp(table_name)
            
            if latest_time:
                # Start from the next hour after latest data
                start_time = latest_time + timedelta(hours=1)
                self.logger.info(f"Incremental download from {start_time} for {table_name}")
                return start_time
            else:
                # No existing data, start from default days back
                start_time = datetime.now() - timedelta(days=default_days_back)
                self.logger.info(f"Full download from {start_time} for {table_name}")
                return start_time
                
        except Exception as e:
            self.logger.error(f"Error determining start time for {table_name}: {e}")
            # Fallback to default
            return datetime.now() - timedelta(days=default_days_back)
    
    def download_and_store_btc_data(self, full_refresh: bool = False) -> Dict[str, int]:
        """
        Download all BTC metrics and store in database
        
        Args:
            full_refresh: If True, downloads all historical data
            
        Returns:
            Dictionary with counts of records downloaded per metric
        """
        results = {}
        start_time = None
        end_time = None
        
        try:
            # Determine time ranges
            if full_refresh:
                # Start from 2021-02-01 as in original code
                start_time = datetime(2021, 2, 1)
                self.logger.info("Performing full data refresh from 2021-02-01")
            else:
                # Get latest timestamp from price data (primary table)
                start_time = self.get_download_start_time('btc_price_data', default_days_back=30)
                self.logger.info(f"Incremental download starting from {start_time}")
            
            end_time = datetime.now()
            
            # Download price data
            self.logger.info("Downloading BTC price data...")
            df_price = self.download_metric(
                "metrics/market/price_usd_close",
                "price",
                start_time,
                end_time
            )
            
            if not df_price.empty:
                self.db.insert_dataframe(df_price, 'btc_price_data', conflict_resolution='update')
                results['price'] = len(df_price)
                self.logger.info(f"Stored {len(df_price)} price records")
            
            # Download liquidation metrics
            liquidation_metrics = [
                ("derivatives/futures_liquidated_volume_long_relative", "long_liquidations_dominance"),
                ("derivatives/futures_liquidated_volume_long_mean", "futures_long_liquidations"),
                ("derivatives/futures_liquidated_volume_short_mean", "futures_short_liquidations")
            ]
            
            liquidation_dfs = []
            for endpoint, metric_name in liquidation_metrics:
                self.logger.info(f"Downloading {metric_name}...")
                df = self.download_metric(f"metrics/{endpoint}", metric_name, start_time, end_time)
                if not df.empty:
                    df = df.set_index('time')
                    liquidation_dfs.append(df)
                    results[metric_name] = len(df)
            
            # Combine liquidation data
            if liquidation_dfs:
                df_liquidations = pd.concat(liquidation_dfs, axis=1, join='outer')
                df_liquidations = df_liquidations.reset_index()
                
                # Handle NaN values as in original code
                self._handle_liquidation_nans(df_liquidations)
                
                # Store in database
                if not df_liquidations.empty:
                    self.db.insert_dataframe(df_liquidations, 'btc_liquidations', conflict_resolution='update')
                    self.logger.info(f"Stored {len(df_liquidations)} liquidation records")
            
            # Log download activity
            total_records = sum(results.values())
            self.db.log_download(
                endpoint="glassnode_bulk",
                start_date=start_time,
                end_date=end_time,
                records=total_records,
                status='success'
            )
            
            self.logger.info(f"Download completed successfully: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in download process: {e}")
            if start_time and end_time:
                self.db.log_download(
                    endpoint="glassnode_bulk",
                    start_date=start_time,
                    end_date=end_time,
                    records=0,
                    status='failed',
                    error=str(e)
                )
            raise
    
    def _handle_liquidation_nans(self, df: pd.DataFrame):
        """Handle NaN values in liquidation data as per original logic"""
        if df.empty:
            return
        
        # Log initial data quality
        initial_rows = len(df)
        
        # Log last valid values for debugging
        for col in ['futures_long_liquidations', 'futures_short_liquidations', 'long_liquidations_dominance']:
            if col in df.columns:
                last_valid_idx = df[col].last_valid_index()
                if last_valid_idx is not None:
                    self.logger.debug(f"Last valid time for {col}: {df.loc[last_valid_idx, 'time']}")
        
        # Fill long_liquidations_dominance NaNs with calculated values
        if all(col in df.columns for col in ['futures_long_liquidations', 'futures_short_liquidations']):
            # Calculate dominance ratio
            total_liquidations = df['futures_long_liquidations'] + df['futures_short_liquidations']
            
            # Avoid division by zero
            replacement_values = df['futures_long_liquidations'] / (total_liquidations + 1e-10)
            
            # Fill NaNs in long_liquidations_dominance
            df['long_liquidations_dominance'] = df['long_liquidations_dominance'].fillna(replacement_values)
            
            filled_count = df['long_liquidations_dominance'].notna().sum() - df['long_liquidations_dominance'].shift(1).notna().sum()
            if filled_count > 0:
                self.logger.info(f"Filled {filled_count} NaN values in long_liquidations_dominance")
        
        # Drop rows with remaining NaN values
        df.dropna(inplace=True)
        
        final_rows = len(df)
        if final_rows < initial_rows:
            self.logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
    
    def verify_data_integrity(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Verify data integrity for a given date range
        
        Returns:
            Dictionary with verification results
        """
        results = {
            'price_data': {},
            'liquidation_data': {},
            'missing_hours': [],
            'data_quality': {}
        }
        
        try:
            # Check price data
            price_df = self.db.get_data_range('btc_price_data', start_date, end_date)
            results['price_data']['count'] = len(price_df)
            results['price_data']['has_nulls'] = price_df['price'].isnull().any() if not price_df.empty else True
            
            # Check liquidation data
            liq_df = self.db.get_data_range('btc_liquidations', start_date, end_date)
            results['liquidation_data']['count'] = len(liq_df)
            
            if not liq_df.empty:
                for col in ['long_liquidations_dominance', 'futures_long_liquidations', 'futures_short_liquidations']:
                    if col in liq_df.columns:
                        results['liquidation_data'][f'{col}_nulls'] = liq_df[col].isnull().sum()
            
            # Check for missing hours
            expected_hours = pd.date_range(start=start_date, end=end_date, freq='H')
            if not price_df.empty:
                actual_hours = pd.to_datetime(price_df['time'])
                missing = set(expected_hours) - set(actual_hours)
                results['missing_hours'] = sorted(list(missing))
                results['data_quality']['completeness'] = len(actual_hours) / len(expected_hours)
            else:
                results['data_quality']['completeness'] = 0.0
            
            # Calculate overall data quality score
            quality_score = 1.0
            if results['price_data'].get('has_nulls', False):
                quality_score -= 0.2
            if results['liquidation_data']['count'] == 0:
                quality_score -= 0.3
            if len(results['missing_hours']) > 0:
                quality_score -= 0.3 * (len(results['missing_hours']) / len(expected_hours))
            
            results['data_quality']['overall_score'] = max(0.0, quality_score)
            
        except Exception as e:
            self.logger.error(f"Error in data integrity verification: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """Get download statistics from the database"""
        try:
            # Get recent download logs
            logs_df = self.db.read_data("""
                SELECT * FROM data_download_logs 
                ORDER BY download_time DESC 
                LIMIT 10
            """)
            
            # Get table statistics
            tables = ['btc_price_data', 'btc_liquidations']
            table_stats = {}
            
            for table in tables:
                info = self.db.get_table_info(table)
                table_stats[table] = info
            
            return {
                'recent_downloads': logs_df.to_dict('records') if not logs_df.empty else [],
                'table_statistics': table_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting download statistics: {e}")
            return {'error': str(e)}


# Convenience functions for command-line usage
def download_latest_data(db_config: Optional[Dict] = None):
    """Download latest data (incremental update)"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_config)
        db.create_tables()  # Ensure tables exist
        
        downloader = BTCDataDownloader(db, logger)
        results = downloader.download_and_store_btc_data(full_refresh=False)
        
        logger.info(f"Latest data download complete: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Latest data download failed: {e}")
        raise


def download_full_history(db_config: Optional[Dict] = None):
    """Download full historical data"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_config)
        db.create_tables()  # Ensure tables exist
        
        downloader = BTCDataDownloader(db, logger)
        results = downloader.download_and_store_btc_data(full_refresh=True)
        
        logger.info(f"Full history download complete: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Full history download failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        download_full_history()
    else:
        download_latest_data()