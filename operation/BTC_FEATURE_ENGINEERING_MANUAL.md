# BTC Feature Engineering Pipeline - Operations Manual

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Database Setup](#database-setup)
5. [Daily Operations](#daily-operations)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)
9. [Performance Optimization](#performance-optimization)
10. [Dashboard Visualization](#dashboard-visualization)
11. [Backup & Recovery](#backup--recovery)

---

## 🏗️ System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    BTC Feature Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │   Glassnode │  │ PostgreSQL   │  │ Interactive     │     │
│  │     API     │  │   Database   │  │  Dashboard      │     │
│  └─────────────┘  └──────────────┘  └─────────────────┘     │
│         │                 │                    │             │
│  ┌──────▼──────┐ ┌────────▼────────┐ ┌─────────▼──────┐     │
│  │ Data        │ │ Feature         │ │ Dash Web App   │     │
│  │ Downloader  │ │ Engineering     │ │ + Flask API    │     │
│  └─────────────┘ └─────────────────┘ └────────────────┘     │
│         │                 │                    │             │
│  ┌──────▼──────┐ ┌────────▼────────┐ ┌─────────▼──────┐     │
│  │ Data        │ │ Database        │ │ Real-time      │     │
│  │ Preprocessor│ │ Manager         │ │ Visualization  │     │
│  └─────────────┘ └─────────────────┘ └────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Incremental Updates**: Only processes new data
- **Batch Processing**: Handles large datasets efficiently  
- **Error Recovery**: Comprehensive error handling and logging
- **Performance**: Optimized queries and indexes
- **Modularity**: Each component can be used independently
- **Real-time Dashboard**: Interactive Plotly visualizations with live data
- **Web Integration**: REST APIs and responsive web interface

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 12+ 
- 4GB+ RAM recommended
- 50GB+ disk space for historical data

### 1. Install Dependencies

```bash
cd feature_engineer
pip install -r requirements.txt

# Install dashboard dependencies
pip install -r ../dashboard/requirements.txt
```

### 2. PostgreSQL Setup

```bash
# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Create database and user
psql postgres
CREATE DATABASE btc_analysis;
CREATE USER btc_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE btc_analysis TO btc_user;
\q
```

### 3. Environment Variables

Create `.env` file:

```bash
# Database Configuration
export DB_HOST=localhost
export DB_PORT=5432  
export DB_NAME=btc_analysis
export DB_USER=btc_user
export DB_PASSWORD=secure_password

# Glassnode API
export GLASSNODE_API_KEY=your_api_key_here

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=btc_pipeline.log
```

### 4. Initialize Database

```bash
python main.py --init-db
```

---

## ⚙️ Configuration

### Configuration File

Create `config/pipeline_config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  database: btc_analysis
  user: btc_user
  password: secure_password

glassnode:
  api_key: "your_api_key_here"
  base_url: "https://api.glassnode.com/v1/"
  requests_per_minute: 10

processing:
  batch_size: 10000
  default_days_back: 30
  max_age_hours: 2

features:
  rpn:
    wavelet: coif3
    level: 6
  bin:
    n_clusters: 9
  signals:
    beta_window: 8
```

### Configuration Validation

```bash
# Create sample config
python -c "
from config.config_manager import ConfigManager
cm = ConfigManager()
cm.create_sample_config('sample_config.yaml')
"

# Validate current config
python -c "
from config.config_manager import ConfigManager
cm = ConfigManager('config/pipeline_config.yaml')
result = cm.validate_config()
print('Valid:', result['valid'])
print('Errors:', result['errors'])
print('Warnings:', result['warnings'])
"
```

---

## 🗄️ Database Setup

### Database Schema

The pipeline creates these tables automatically:

```sql
-- Raw Data Tables
btc_price_data          -- BTC price data from Glassnode
btc_liquidations        -- Liquidation data

-- Processed Data
btc_processed_data      -- Preprocessed and normalized data

-- Feature Tables  
btc_features_rpn        -- Risk Priority Number features
btc_features_bin        -- Binning and clustering features
btc_features_dominance  -- Market dominance features
btc_features_signals    -- Trading signals

-- Metadata Tables
feature_engineering_runs -- Processing run logs
data_download_logs      -- Download activity logs
```

### Manual Database Operations

```bash
# Connect to database
psql -h localhost -U btc_user -d btc_analysis

# Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Check data ranges
SELECT 
    'btc_price_data' as table_name,
    MIN(time) as earliest,
    MAX(time) as latest,
    COUNT(*) as records
FROM btc_price_data
UNION ALL
SELECT 
    'btc_features_signals',
    MIN(time),
    MAX(time), 
    COUNT(*)
FROM btc_features_signals;
```

---

## 📅 Daily Operations

### 1. Standard Pipeline Run

```bash
# Full pipeline (recommended for daily runs)
python main.py --full-pipeline

# Check status first
python main.py --status

# Download only (if needed)
python main.py --download

# Process features only  
python main.py --features

# Start interactive dashboard
python main.py --dashboard

# Or use the unified script
python ../run_pipeline_with_dashboard.py run
```

### 2. Scheduled Operations

#### Using Cron (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Add hourly updates
0 * * * * cd /path/to/feature_engineer && python main.py --full-pipeline >> /var/log/btc_pipeline.log 2>&1

# Add daily full refresh (optional)  
0 2 * * * cd /path/to/feature_engineer && python main.py --full-pipeline --full-refresh >> /var/log/btc_pipeline.log 2>&1
```

#### Using Built-in Scheduler

```bash
# Run scheduled updates every hour
python main.py --schedule --interval 3600

# Run in background
nohup python main.py --schedule --interval 3600 > scheduler.log 2>&1 &
```

### 3. Data Export Operations

```bash
# Export training data
python main.py --export --output training_data.csv --feature-groups rpn bin dominance

# Export specific date range
python main.py --export --start-date 2024-01-01 --end-date 2024-12-31 --output 2024_data.csv

# Export all data types
python -c "
from data_reader.reader import BTCDataReader
from db.database_manager import DatabaseManager

db = DatabaseManager()
reader = BTCDataReader(db)
reader.export_to_csv('exports/', start_time=None, end_time=None)
"
```

---

## 📊 Monitoring & Maintenance

### 1. Health Checks

#### Daily Health Check Script

```bash
#!/bin/bash
# health_check.sh

echo "=== BTC Pipeline Health Check $(date) ==="

# Check database connectivity
python -c "
from db.database_manager import DatabaseManager
try:
    db = DatabaseManager()
    with db.get_connection() as conn:
        print('✓ Database connection: OK')
except Exception as e:
    print(f'✗ Database connection: FAILED - {e}')
"

# Check data freshness
python -c "
from db.database_manager import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()
latest = db.get_latest_timestamp('btc_price_data')
if latest:
    age = datetime.now() - latest.replace(tzinfo=None)
    if age < timedelta(hours=6):
        print(f'✓ Data freshness: OK (last update: {latest})')
    else:
        print(f'⚠ Data freshness: STALE (last update: {latest})')
else:
    print('✗ Data freshness: NO DATA')
"

# Check pipeline status
python main.py --status | grep -E "(error|ERROR)" && echo "✗ Errors found in status" || echo "✓ Status check: OK"

echo "=== Health Check Complete ==="
```

#### Set up health monitoring

```bash
chmod +x health_check.sh

# Add to cron for daily health checks
echo "0 8 * * * /path/to/health_check.sh >> /var/log/health_check.log 2>&1" | crontab -
```

### 2. Performance Monitoring

#### Database Performance

```sql
-- Check table sizes and growth
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename)) as size,
    pg_total_relation_size(tablename) as size_bytes
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY size_bytes DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

#### Application Performance

```bash
# Check log for performance issues
tail -n 100 btc_pipeline.log | grep -E "(ERROR|WARNING|slow|timeout)"

# Monitor memory usage during processing
python -c "
import psutil
import time
from main import BTCFeaturePipeline

process = psutil.Process()
print(f'Memory before: {process.memory_info().rss / 1024 / 1024:.1f} MB')

# Run pipeline monitoring
# pipeline = BTCFeaturePipeline()
# Add your monitoring code here

print(f'Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

### 3. Log Management

#### Log Rotation Setup

```bash
# Create logrotate config
sudo tee /etc/logrotate.d/btc-pipeline << EOF
/path/to/feature_engineer/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 user user
}
EOF
```

#### Log Analysis

```bash
# Error analysis
grep -E "(ERROR|FAILED)" btc_pipeline.log | tail -20

# Performance analysis  
grep -E "(completed|duration)" btc_pipeline.log | tail -10

# Signal analysis
grep "signal" btc_pipeline.log | grep -v "DEBUG" | tail -20
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Database Connection Issues

**Problem**: `psycopg2.OperationalError: could not connect to server`

**Solutions**:
```bash
# Check PostgreSQL is running
brew services list | grep postgresql
sudo systemctl status postgresql  # Linux

# Check connection parameters
psql -h localhost -U btc_user -d btc_analysis

# Reset connection pool
python -c "
from db.database_manager import DatabaseManager
db = DatabaseManager()
# Connection pool will be recreated
"
```

#### 2. API Rate Limiting

**Problem**: `HTTP 429: Too Many Requests`

**Solutions**:
```yaml
# Reduce request rate in config
glassnode:
  requests_per_minute: 5  # Reduce from 10

# Or increase delay between requests
processing:
  api_delay_seconds: 15
```

#### 3. Memory Issues

**Problem**: `MemoryError` during feature processing

**Solutions**:
```yaml
# Reduce batch size
processing:
  batch_size: 5000  # Reduce from 10000

# Or process smaller date ranges
python main.py --features --start-date 2024-01-01 --end-date 2024-01-31
```

#### 4. Missing Data

**Problem**: Gaps in processed data

**Solutions**:
```bash
# Check data integrity
python -c "
from data_download.downloader import BTCDataDownloader
from db.database_manager import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()
downloader = BTCDataDownloader(db)

# Check last 7 days
start = datetime.now() - timedelta(days=7)
end = datetime.now()
integrity = downloader.verify_data_integrity(start, end)
print('Integrity report:', integrity)
"

# Reprocess specific date range
python main.py --download --start-date 2024-01-01 --end-date 2024-01-31
python main.py --preprocess --start-date 2024-01-01 --end-date 2024-01-31
python main.py --features --start-date 2024-01-01 --end-date 2024-01-31
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with detailed output
python main.py --full-pipeline 2>&1 | tee debug.log

# Check specific component
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from feature_engineering.feature_engineer import FeatureEngineer
from db.database_manager import DatabaseManager

db = DatabaseManager()
engineer = FeatureEngineer(db)
# Add debug code here
"
```

---

## 📚 API Reference

### Python API Usage

#### Basic Pipeline Usage

```python
from feature_engineer import BTCFeaturePipeline

# Initialize pipeline
pipeline = BTCFeaturePipeline('config/pipeline_config.yaml')

# Run full pipeline
results = pipeline.run_full_pipeline()
print(f"Pipeline completed: {results}")

# Get current status
status = pipeline.get_pipeline_status()
print(f"Database status: {status}")
```

#### Individual Components

```python
from db.database_manager import DatabaseManager
from data_reader.reader import BTCDataReader
from datetime import datetime, timedelta

# Initialize components
db = DatabaseManager()
reader = BTCDataReader(db)

# Get latest features
end_time = datetime.now()
start_time = end_time - timedelta(days=7)
features = reader.get_all_features(start_time, end_time)
print(f"Retrieved {len(features)} feature records")

# Get trading signals
signals = reader.get_latest_signals(hours=24)
print(f"Found {len(signals)} signals in last 24 hours")

# Export training data
training_data = reader.get_training_data(
    feature_columns=['risk_priority_number', 'bin_index', 'dominance'],
    target_column='signal_class',
    min_hours_between_signals=6
)
training_data.to_csv('training_data.csv', index=False)
```

#### Advanced Data Access

```python
# Custom queries
query = """
    SELECT 
        p.time,
        p.price,
        r.risk_priority_number,
        s.signal_class
    FROM btc_processed_data p
    JOIN btc_features_rpn r ON p.time = r.time
    JOIN btc_features_signals s ON p.time = s.time
    WHERE s.signal_class != 'Neutral'
    AND p.time >= NOW() - INTERVAL '7 days'
    ORDER BY p.time DESC
"""

results = db.read_data(query)
print(f"Found {len(results)} trading signals")

# Performance metrics
metrics = reader.get_performance_metrics(hours=24*30)  # Last 30 days
print("Signal performance:")
for _, row in metrics.iterrows():
    print(f"  {row['signal_class']}: {row['win_rate_24h']:.1%} win rate")
```

---

## ⚡ Performance Optimization

### Database Optimization

#### Index Management

```sql
-- Create additional indexes for performance
CREATE INDEX CONCURRENTLY idx_btc_price_data_time_price ON btc_price_data(time, price);
CREATE INDEX CONCURRENTLY idx_btc_features_signals_class_time ON btc_features_signals(signal_class, time);
CREATE INDEX CONCURRENTLY idx_btc_features_rpn_rpn_time ON btc_features_rpn(risk_priority_number, time);

-- Monitor index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- Remove unused indexes
DROP INDEX IF EXISTS unused_index_name;
```

#### Query Optimization

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM btc_latest_features 
WHERE time >= NOW() - INTERVAL '24 hours';

-- Update table statistics
ANALYZE btc_price_data;
ANALYZE btc_features_signals;
ANALYZE btc_features_rpn;

-- Vacuum tables
VACUUM ANALYZE btc_price_data;
VACUUM ANALYZE btc_features_signals;
```

#### Connection Pool Tuning

```python
# Optimize connection pool
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'btc_analysis', 
    'user': 'btc_user',
    'password': 'password',
    # Pool settings
    'minconn': 5,      # Minimum connections
    'maxconn': 20,     # Maximum connections 
    'keepalives_idle': 600,
    'keepalives_interval': 30,
    'keepalives_count': 3
}

db = DatabaseManager(db_config)
```

### Application Optimization

#### Memory Optimization

```python
# Process in smaller batches
pipeline_config = {
    'processing': {
        'batch_size': 5000,  # Smaller batches
        'chunk_size': 1000,  # Process chunks within batches
        'memory_limit_mb': 2048
    }
}

# Use generators for large datasets
def process_in_chunks(df, chunk_size=1000):
    for i in range(0, len(df), chunk_size):
        yield df[i:i+chunk_size]

# Example usage
large_df = reader.get_all_features()
for chunk in process_in_chunks(large_df):
    # Process chunk
    processed_chunk = process_features(chunk)
    # Save results
    save_chunk(processed_chunk)
```

#### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parallel_feature_processing(df, n_workers=4):
    """Process features in parallel"""
    
    # Split dataframe
    chunks = np.array_split(df, n_workers)
    
    def process_chunk(chunk):
        return feature_engineer.process_rpn_features(chunk)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    return pd.concat(results, ignore_index=True)
```

---

## 📊 Dashboard Visualization

### Overview

The BTC Feature Engineering Pipeline includes an integrated web dashboard for real-time visualization and monitoring of:

- **Interactive Charts**: Price movements with risk priority bins
- **Signal Analysis**: Buy/sell signals with performance tracking
- **Feature Visualization**: RPN, dominance, and liquidation patterns
- **Performance Metrics**: Signal win rates and return statistics
- **System Monitoring**: Database status and pipeline health

### Dashboard Components

#### 1. Dash Web Application (Recommended)
- **Port**: 8050 (default)
- **Technology**: Python Dash + Plotly
- **Features**: Fully interactive with real-time updates

#### 2. Flask REST API
- **Port**: 5000 (default)  
- **Endpoints**: `/api/features`, `/api/plots/<type>`, `/api/signals`
- **Usage**: For custom frontends or API integrations

#### 3. React Frontend (Optional)
- **Port**: 3000 (when used)
- **Technology**: React + TypeScript
- **Features**: Modern responsive interface

### Quick Start

#### Option 1: Integrated Pipeline + Dashboard
```bash
# Run pipeline once then start dashboard
python run_pipeline_with_dashboard.py run

# Run scheduled pipeline with persistent dashboard  
python run_pipeline_with_dashboard.py schedule --interval 3600

# Dashboard only (no pipeline execution)
python run_pipeline_with_dashboard.py dashboard
```

#### Option 2: Pipeline Built-in Dashboard
```bash
# From feature_engineer directory
python main.py --dashboard --port 8050

# With custom configuration
python main.py --dashboard --config my_config.yaml --no-browser
```

#### Option 3: Standalone Dashboard
```bash
# From dashboard directory
python dash_app.py

# Or Flask API
python backend.py
```

### Dashboard Features

#### Overview Tab
- **Price Chart**: BTC price colored by risk priority bins
- **RPN Analysis**: Risk Priority Number with dominance indicators
- **Statistics Cards**: Real-time database statistics

#### Signals Tab  
- **Signal Visualization**: Buy/sell signals on price chart
- **Latest Signals Table**: Recent signals with classifications
- **Signal Strength**: Visual indicators for signal confidence

#### Analysis Tab
- **Liquidations**: Futures long vs short liquidation patterns
- **Dominance Status**: Market dominance over time
- **Correlation Analysis**: Relationship between signals and outcomes

#### Performance Tab
- **Win Rates**: Signal performance at 1h, 6h, 24h horizons
- **Return Analysis**: Average returns by signal type
- **Success Metrics**: Statistical performance indicators

### Time Range Selection

Choose from predefined ranges or custom dates:
- **1 Hour**: Last hour of data
- **6 Hours**: Last 6 hours
- **24 Hours**: Last day (default)
- **7 Days**: Last week
- **30 Days**: Last month
- **90 Days**: Last quarter
- **Custom**: Specify exact date range

### Configuration

#### Dashboard Settings
```yaml
# config/dashboard_config.yaml
dashboard:
  port: 8050
  host: "0.0.0.0"
  debug: false
  auto_refresh_interval: 300  # 5 minutes
  
plots:
  height: 500
  template: "plotly_dark"
  max_data_points: 10000
  
time_ranges:
  default: "24h"
  available: ["1h", "6h", "24h", "7d", "30d", "90d", "custom"]
```

#### API Configuration
```yaml
api:
  cors_enabled: true
  rate_limit: 100  # requests per minute
  cache_timeout: 60  # seconds
  
export:
  max_records: 50000
  supported_formats: ["json", "csv"]
```

### Development and Customization

#### Adding Custom Plots

1. **Create visualization method**:
```python
# In dashboard/backend.py - PlotlyVisualizer class
def create_custom_plot(self) -> Dict:
    fig = go.Figure()
    
    # Add your visualization logic
    fig.add_trace(go.Scatter(
        x=self.data['time'],
        y=self.data['your_feature'],
        name='Custom Feature'
    ))
    
    return fig.to_dict()
```

2. **Add to dashboard layout**:
```python
# In dashboard/dash_app.py
dcc.Graph(
    figure=viz.create_custom_plot(),
    style={'height': '500px'}
)
```

3. **Add API endpoint**:
```python
# In dashboard/backend.py
@app.route('/api/plots/custom', methods=['GET'])
def get_custom_plot():
    # Implementation
    return jsonify(plot_data)
```

#### Extending Data Sources

```python
# Custom data reader integration
class CustomDataReader(BTCDataReader):
    def get_custom_features(self, start_time, end_time):
        query = """
        SELECT time, custom_feature1, custom_feature2
        FROM custom_features_table
        WHERE time BETWEEN %s AND %s
        ORDER BY time
        """
        return self.db.read_data(query, (start_time, end_time))
```

### Production Deployment

#### Using Gunicorn (Recommended)
```bash
# Install gunicorn
pip install gunicorn

# Run dashboard
cd dashboard
gunicorn dash_app:server -b 0.0.0.0:8050 -w 4

# With environment variables
DB_HOST=prod-db DB_PASSWORD=secure_pass gunicorn dash_app:server -b 0.0.0.0:8050
```

#### Using Docker
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8050

CMD ["gunicorn", "dashboard.dash_app:server", "-b", "0.0.0.0:8050"]
```

#### Nginx Reverse Proxy
```nginx
# /etc/nginx/sites-available/btc-dashboard
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring and Alerts

#### Dashboard Health Check
```bash
# Check dashboard availability
curl -f http://localhost:8050/api/stats || echo "Dashboard down"

# Check data freshness
python -c "
import requests
r = requests.get('http://localhost:8050/api/stats')
data = r.json()
print('Latest data:', data.get('btc_price_data', {}).get('latest', 'N/A'))
"
```

#### Set Up Alerts
```bash
# Add to crontab for monitoring
*/5 * * * * curl -f http://localhost:8050/api/stats >/dev/null 2>&1 || echo "Dashboard down" | mail admin@company.com
```

### Troubleshooting Dashboard

#### Common Issues

**Dashboard won't start**:
```bash
# Check dependencies
python -c "import dash, plotly, flask; print('Dependencies OK')"

# Check port availability
netstat -an | grep 8050

# Check database connection
python -c "from db.database_manager import DatabaseManager; DatabaseManager()"
```

**No data displaying**:
```bash
# Verify data exists
python -c "
from data_reader.reader import BTCDataReader
from db.database_manager import DatabaseManager
reader = BTCDataReader(DatabaseManager())
df = reader.get_all_features()
print(f'Found {len(df)} records')
"
```

**Performance issues**:
```bash
# Reduce time range or data points
# Check database query performance
# Add database indexes if needed
```

#### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run dashboard with debug
python dash_app.py --debug

# Check browser console for JavaScript errors
```

### API Reference

#### Available Endpoints

- `GET /api/features?hours=24` - Get feature data
- `GET /api/plots/price_color?hours=168` - Get price plot data
- `GET /api/latest_signals?hours=24` - Get recent signals
- `GET /api/performance?hours=168` - Get performance metrics
- `GET /api/stats` - Get database statistics

#### Response Format
```json
{
  "data": [...],
  "metadata": {
    "start_time": "2024-01-01T00:00:00",
    "end_time": "2024-01-02T00:00:00", 
    "record_count": 1440
  }
}
```

---

## 💾 Backup & Recovery

### Database Backup

#### Automated Backup Script

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backup/btc_pipeline"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="btc_analysis_backup_${DATE}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
pg_dump -h localhost -U btc_user -d btc_analysis > $BACKUP_DIR/$BACKUP_FILE

# Compress backup
gzip $BACKUP_DIR/$BACKUP_FILE

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/${BACKUP_FILE}.gz"
```

#### Schedule Backups

```bash
# Daily backup at 3 AM
echo "0 3 * * * /path/to/backup_database.sh >> /var/log/backup.log 2>&1" | crontab -
```

### Data Recovery

#### Full Database Restore

```bash
# Stop pipeline processes
sudo systemctl stop btc-pipeline

# Drop and recreate database
dropdb btc_analysis
createdb btc_analysis

# Restore from backup
gunzip -c /backup/btc_pipeline/btc_analysis_backup_20240101_030000.sql.gz | psql -h localhost -U btc_user -d btc_analysis

# Restart pipeline
sudo systemctl start btc-pipeline
```

#### Partial Data Recovery

```bash
# Restore specific table
pg_restore -h localhost -U btc_user -d btc_analysis -t btc_features_signals backup_file.sql

# Reprocess specific date range
python main.py --features --start-date 2024-01-01 --end-date 2024-01-31
```

### Configuration Backup

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ *.yaml *.json

# Backup to remote location
rsync -av config/ user@backup-server:/backup/btc_pipeline/config/
```

---

## 📞 Support & Contacts

### Emergency Procedures

1. **Pipeline Down**: Check database connectivity, restart services
2. **Data Loss**: Restore from latest backup, reprocess missing data  
3. **Performance Issues**: Check memory usage, reduce batch sizes
4. **API Issues**: Check Glassnode status, adjust rate limits

### Monitoring Alerts

Set up alerts for:
- Pipeline failures (email/Slack)
- Data age > 6 hours
- Database size > 80% capacity
- Error rate > 5%

### Log Locations

- Main pipeline: `btc_pipeline.log`
- Database: `/var/log/postgresql/postgresql.log`
- System: `/var/log/syslog`
- Health checks: `/var/log/health_check.log`

---

*Last updated: January 2025*  
*Version: 2.0.0*