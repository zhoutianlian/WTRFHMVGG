import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime
import json
import uuid


class DatabaseManager:
    """PostgreSQL database manager for BTC feature engineering pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database connection pool
        
        Args:
            config: Database configuration dictionary. If None, uses environment variables.
        """
        if config is None:
            config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'btc_analysis'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres')
            }
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection pool
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                **config
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
        
    def __del__(self):
        """Close connection pool when object is destroyed"""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections from pool"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Database operation error: {e}")
                raise
            finally:
                cursor.close()
    
    def execute_sql_file(self, filepath: str):
        """Execute SQL file"""
        with open(filepath, 'r') as f:
            sql = f.read()
        
        with self.get_cursor() as cursor:
            cursor.execute(sql)
            self.logger.info(f"Executed SQL file: {filepath}")
    
    def create_tables(self):
        """Create all necessary tables if they don't exist"""
        schema_sql = """
        -- Raw Data Tables
        CREATE TABLE IF NOT EXISTS btc_price_data (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            price DECIMAL(20, 8) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_price_time ON btc_price_data(time);

        CREATE TABLE IF NOT EXISTS btc_liquidations (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            long_liquidations_dominance DECIMAL(10, 8),
            futures_long_liquidations DECIMAL(20, 8),
            futures_short_liquidations DECIMAL(20, 8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_liquidations_time ON btc_liquidations(time);

        -- Processed Data Tables
        CREATE TABLE IF NOT EXISTS btc_processed_data (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            price DECIMAL(20, 8) NOT NULL,
            lld_normal DECIMAL(10, 8),
            fll_normal DECIMAL(20, 8),
            fsl_normal DECIMAL(20, 8),
            diff_ls_normal DECIMAL(20, 8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_processed_time ON btc_processed_data(time);

        -- Feature Tables
        CREATE TABLE IF NOT EXISTS btc_features_rpn (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            price DECIMAL(20, 8) NOT NULL,
            risk_priority_number DECIMAL(10, 8),
            lld_cwt_kf DECIMAL(10, 8),
            fll_cwt_kf DECIMAL(20, 8),
            fsl_cwt_kf DECIMAL(20, 8),
            diff_ls_cwt_kf DECIMAL(20, 8),
            lld_cwt_kf_smooth DECIMAL(10, 8),
            diff_ls_smooth DECIMAL(20, 8),
            corr_case VARCHAR(10),
            delta_fll DECIMAL(20, 8),
            delta_fsl DECIMAL(20, 8),
            is_rpn_extreme INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_features_rpn_time ON btc_features_rpn(time);

        CREATE TABLE IF NOT EXISTS btc_features_bin (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            bin_index INTEGER NOT NULL,
            bin_momentum DECIMAL(10, 8),
            bin_turn INTEGER,
            bin_mm_direction INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_features_bin_time ON btc_features_bin(time);

        CREATE TABLE IF NOT EXISTS btc_features_dominance (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            dominance INTEGER,
            dominance_last INTEGER,
            dominance_duration INTEGER,
            dominance_duration_total INTEGER,
            dominance_time INTEGER,
            dominance_class VARCHAR(20),
            is_keep INTEGER,
            is_strengthen INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_features_dom_time ON btc_features_dominance(time);

        CREATE TABLE IF NOT EXISTS btc_features_signals (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            hit_ceiling_bottom INTEGER,
            reverse_ceiling_bottom INTEGER,
            signal_l1 INTEGER,
            signal_l2 INTEGER,
            signal_l3 INTEGER,
            signal_class VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_features_signals_time ON btc_features_signals(time);

        CREATE TABLE IF NOT EXISTS btc_features_advanced (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL UNIQUE,
            -- KAMA Features
            fll_kama DECIMAL(20, 8),
            fsl_kama DECIMAL(20, 8),
            price_kama DECIMAL(20, 8),
            -- Spike Ratio Features
            fll_spike_kama DECIMAL(10, 8),
            fll_spike_kama_ma DECIMAL(20, 8),
            fll_spike_kama_strength DECIMAL(10, 8),
            fll_spike_kama_direction INTEGER,
            fll_spike_kama_rolling_mean DECIMAL(10, 8),
            fll_spike_kama_rolling_std DECIMAL(10, 8),
            fll_spike_kama_zscore DECIMAL(10, 8),
            fll_spike_kama_extreme INTEGER,
            fsl_spike_kama DECIMAL(10, 8),
            fsl_spike_kama_ma DECIMAL(20, 8),
            fsl_spike_kama_strength DECIMAL(10, 8),
            fsl_spike_kama_direction INTEGER,
            fsl_spike_kama_rolling_mean DECIMAL(10, 8),
            fsl_spike_kama_rolling_std DECIMAL(10, 8),
            fsl_spike_kama_zscore DECIMAL(10, 8),
            fsl_spike_kama_extreme INTEGER,
            price_spike_ema DECIMAL(10, 8),
            price_spike_ema_ma DECIMAL(20, 8),
            price_spike_ema_strength DECIMAL(10, 8),
            price_spike_ema_direction INTEGER,
            price_spike_ema_rolling_mean DECIMAL(10, 8),
            price_spike_ema_rolling_std DECIMAL(10, 8),
            price_spike_ema_zscore DECIMAL(10, 8),
            price_spike_ema_extreme INTEGER,
            -- ROC Dynamics Features
            fll_velocity DECIMAL(20, 8),
            fll_acceleration DECIMAL(20, 8),
            price_velocity DECIMAL(20, 8),
            price_acceleration DECIMAL(20, 8),
            -- Momentum Indicators
            price_rsi DECIMAL(10, 8),
            price_stoch_k DECIMAL(10, 8),
            price_stoch_d DECIMAL(10, 8),
            -- MACD Features
            price_macd DECIMAL(20, 8),
            price_macd_signal DECIMAL(20, 8),
            price_macd_histogram DECIMAL(20, 8),
            -- Kalman Slope Features
            kalman_slope DECIMAL(20, 8),
            kalman_slope_filtered_price DECIMAL(20, 8),
            kalman_slope_innovation DECIMAL(20, 8),
            kalman_slope_smooth DECIMAL(20, 8),
            kalman_slope_direction INTEGER,
            kalman_slope_acceleration DECIMAL(20, 8),
            kalman_slope_volatility DECIMAL(10, 8),
            -- Statistical Moments
            price_skewness DECIMAL(10, 8),
            price_skew_signal INTEGER,
            price_kurtosis DECIMAL(10, 8),
            price_excess_kurtosis DECIMAL(10, 8),
            price_fat_tails INTEGER,
            price_jarque_bera DECIMAL(10, 8),
            price_non_normal INTEGER,
            price_cv DECIMAL(10, 8),
            -- Volatility Features
            vol_adaptive DECIMAL(10, 8),
            vol_garch DECIMAL(10, 8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_btc_features_advanced_time ON btc_features_advanced(time);

        -- Metadata Tables
        CREATE TABLE IF NOT EXISTS feature_engineering_runs (
            id SERIAL PRIMARY KEY,
            run_id UUID DEFAULT gen_random_uuid(),
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            status VARCHAR(20) DEFAULT 'running',
            error_message TEXT,
            records_processed INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS data_download_logs (
            id SERIAL PRIMARY KEY,
            download_time TIMESTAMP NOT NULL,
            endpoint VARCHAR(255),
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            records_downloaded INTEGER,
            status VARCHAR(20),
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Views for Web Display
        CREATE OR REPLACE VIEW btc_latest_features AS
        SELECT 
            p.time,
            p.price,
            r.risk_priority_number,
            b.bin_index,
            d.dominance,
            d.dominance_class,
            s.signal_class,
            s.hit_ceiling_bottom,
            s.reverse_ceiling_bottom,
            a.price_kama,
            a.kalman_slope,
            a.price_rsi,
            a.vol_adaptive
        FROM btc_processed_data p
        LEFT JOIN btc_features_rpn r ON p.time = r.time
        LEFT JOIN btc_features_bin b ON p.time = b.time
        LEFT JOIN btc_features_dominance d ON p.time = d.time
        LEFT JOIN btc_features_signals s ON p.time = s.time
        LEFT JOIN btc_features_advanced a ON p.time = a.time
        ORDER BY p.time DESC;
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(schema_sql)
            self.logger.info("Database schema created/updated successfully")
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                        conflict_resolution: str = 'update'):
        """
        Insert DataFrame into database table with conflict resolution
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            conflict_resolution: 'update', 'ignore', or 'raise'
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame provided for table {table_name}")
            return
        
        # Prepare data
        columns = df.columns.tolist()
        values = df.values.tolist()
        
        # Build insert query
        placeholders = ','.join(['%s'] * len(columns))
        column_names = ','.join(columns)
        
        if conflict_resolution == 'update':
            # Assume 'time' is the unique constraint
            update_cols = [f"{col} = EXCLUDED.{col}" for col in columns if col != 'time']
            update_clause = ', '.join(update_cols)
            query = f"""
                INSERT INTO {table_name} ({column_names}) 
                VALUES ({placeholders})
                ON CONFLICT (time) DO UPDATE SET {update_clause}
            """
        elif conflict_resolution == 'ignore':
            query = f"""
                INSERT INTO {table_name} ({column_names}) 
                VALUES ({placeholders})
                ON CONFLICT (time) DO NOTHING
            """
        else:
            query = f"""
                INSERT INTO {table_name} ({column_names}) 
                VALUES ({placeholders})
            """
        
        with self.get_cursor() as cursor:
            execute_batch(cursor, query, values, page_size=1000)
            self.logger.info(f"Inserted {len(df)} records into {table_name}")
    
    def read_data(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Read data from database into DataFrame
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        with self.get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)
            return df
    
    def get_latest_timestamp(self, table_name: str, time_column: str = 'time') -> Optional[datetime]:
        """Get the latest timestamp from a table"""
        query = f"SELECT MAX({time_column}) as max_time FROM {table_name}"
        
        with self.get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            return result['max_time'] if result and result['max_time'] else None
    
    def get_data_range(self, table_name: str, start_time: Optional[datetime] = None, 
                      end_time: Optional[datetime] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get data within a time range
        
        Args:
            table_name: Table to query
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            columns: List of columns to retrieve (None for all)
            
        Returns:
            DataFrame with requested data
        """
        column_str = '*' if columns is None else ', '.join(columns)
        query = f"SELECT {column_str} FROM {table_name} WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND time >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND time <= %s"
            params.append(end_time)
        
        query += " ORDER BY time"
        
        return self.read_data(query, tuple(params) if params else None)
    
    def log_download(self, endpoint: str, start_date: datetime, end_date: datetime, 
                    records: int, status: str = 'success', error: Optional[str] = None):
        """Log data download activity"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO data_download_logs 
                (download_time, endpoint, start_date, end_date, records_downloaded, status, error_message)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s)
            """, (endpoint, start_date, end_date, records, status, error))
    
    def start_feature_run(self) -> str:
        """Start a new feature engineering run and return run_id"""
        with self.get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                INSERT INTO feature_engineering_runs (start_time, status)
                VALUES (NOW(), 'running')
                RETURNING run_id::text
            """)
            return cursor.fetchone()['run_id']
    
    def end_feature_run(self, run_id: str, records: int, status: str = 'success', error: Optional[str] = None):
        """End a feature engineering run"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE feature_engineering_runs 
                SET end_time = NOW(), status = %s, records_processed = %s, error_message = %s
                WHERE run_id = %s::uuid
            """, (status, records, error, run_id))
    
    def get_feature_view(self, limit: int = 1000) -> pd.DataFrame:
        """Get combined feature view for web display"""
        query = f"""
            SELECT 
                p.time,
                p.price,
                r.risk_priority_number,
                b.bin_index,
                d.dominance,
                d.dominance_class,
                s.signal_class,
                s.hit_ceiling_bottom,
                s.reverse_ceiling_bottom
            FROM btc_processed_data p
            LEFT JOIN btc_features_rpn r ON p.time = r.time
            LEFT JOIN btc_features_bin b ON p.time = b.time
            LEFT JOIN btc_features_dominance d ON p.time = d.time
            LEFT JOIN btc_features_signals s ON p.time = s.time
            ORDER BY p.time DESC
            LIMIT %s
        """
        return self.read_data(query, (limit,))
    
    def vacuum_analyze(self, table_name: Optional[str] = None):
        """Run VACUUM ANALYZE on table(s) for performance"""
        with self.get_connection() as conn:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cursor:
                if table_name:
                    cursor.execute(f"VACUUM ANALYZE {table_name}")
                    self.logger.info(f"Vacuumed table: {table_name}")
                else:
                    cursor.execute("VACUUM ANALYZE")
                    self.logger.info("Vacuumed all tables")
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information including size and row count"""
        query = """
            SELECT 
                schemaname,
                tablename,
                attname,
                typename,
                avg_width,
                n_distinct,
                correlation
            FROM pg_stats 
            WHERE tablename = %s
        """
        
        stats_df = self.read_data(query, (table_name,))
        
        # Get table size
        size_query = """
            SELECT 
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE tablename = %s
        """
        
        size_df = self.read_data(size_query, (table_name,))
        
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        count_df = self.read_data(count_query)
        
        return {
            'table_name': table_name,
            'row_count': count_df['row_count'].iloc[0] if not count_df.empty else 0,
            'size': size_df['size'].iloc[0] if not size_df.empty else 'Unknown',
            'size_bytes': size_df['size_bytes'].iloc[0] if not size_df.empty else 0,
            'column_stats': stats_df.to_dict('records') if not stats_df.empty else []
        }
    
    def cleanup_old_data(self, table_name: str, days_to_keep: int = 365):
        """Remove data older than specified days"""
        with self.get_cursor() as cursor:
            cursor.execute(f"""
                DELETE FROM {table_name} 
                WHERE time < NOW() - INTERVAL '{days_to_keep} days'
            """)
            deleted_rows = cursor.rowcount
            self.logger.info(f"Cleaned up {deleted_rows} old records from {table_name}")
            return deleted_rows