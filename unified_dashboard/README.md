# 🔬 BTC Advanced Feature Dashboard

A unified dashboard system that integrates all visualization functions from different modules into one cohesive website with separate pages for each feature category.

## 🎯 Features

### **Integrated Visualization System**
Combines visualization functions from:
- `dashboard/` - Existing Dash apps and Flask backend
- `feature_engineer/advanced_visualization/` - Professional visualizations  
- `feature_engineer/advanced_features/` - Statistical analysis
- **Into one unified dashboard**

### **Feature Categories**
- **🎯 RPN Features**: Risk Priority Number analysis with CWT detrending, Kalman filtering
- **📦 Binning Features**: K-means clustering and momentum analysis
- **👑 Dominance Features**: Market dominance detection and regime classification
- **📡 Signal Features**: Trading signals with ceiling/bottom detection
- **🚀 Advanced Features**: Technical indicators and volatility models
- **📈 Statistical Analysis**: Distribution fitting, normality tests, risk metrics

### **Professional UI/UX**
- **Dark Financial Theme**: Professional dark theme optimized for trading
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Real-time Updates**: Auto-refresh capabilities
- **Navigation System**: Clean navigation between feature categories

## 🚀 Quick Start

### **Run the Dashboard**
```bash
# Simple start (default port 8050)
python unified_dashboard/run_dashboard.py

# Custom port and debug mode
python unified_dashboard/run_dashboard.py --port 8080 --debug

# External access
python unified_dashboard/run_dashboard.py --host 0.0.0.0 --port 8050
```

### **Access the Dashboard**
```
📊 Overview: http://localhost:8050/
🎯 RPN Features: http://localhost:8050/rpn
📦 Binning Features: http://localhost:8050/binning
👑 Dominance Features: http://localhost:8050/dominance
📡 Signal Features: http://localhost:8050/signals
🚀 Advanced Features: http://localhost:8050/advanced
📈 Statistical Analysis: http://localhost:8050/statistical
🔗 API Documentation: http://localhost:8050/api
```

## 📋 Installation

### **Dependencies**
```bash
pip install dash plotly pandas numpy scipy
pip install psycopg2-binary  # For PostgreSQL
```

### **Database Configuration**
```bash
# Using environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=btc_features
export DB_USER=your_user
export DB_PASSWORD=your_password

# Or using command line options
python run_dashboard.py --db-host localhost --db-port 5432 --db-name btc_features
```

## 🏗️ Architecture

### **Core Components**

#### **VisualizationEngine** (`visualization_engine.py`)
- Unified interface to all visualization functions
- Integrates RPNVisualizer, BinningVisualizer, DominanceVisualizer, SignalVisualizer
- Handles data retrieval and chart generation
- Provides fallback mechanisms for missing data

#### **Feature Pages** (`feature_pages.py`)
- Individual page implementations for each feature category
- Consistent layout and interaction patterns
- Real-time data updates with time range selection
- Category-specific metrics and insights

#### **Components** (`components.py`)
- Reusable UI components (NavigationBar, TimeRangeSelector, MetricCard)
- Professional styling and responsive design
- Consistent interaction patterns

#### **Main Dashboard** (`main_dashboard.py`)
- Main application orchestrator
- URL routing and page management
- Professional styling with CSS customizations

### **Integration Strategy**

```python
# Existing visualization modules are imported and wrapped
from feature_engineer.advanced_visualization.rpn_visualizations import RPNVisualizer
from feature_engineer.advanced_visualization.binning_visualizations import BinningVisualizer
from dashboard.backend import PlotlyVisualizer
from feature_engineer.advanced_features.statistical_analyzer import FeatureStatisticalAnalyzer

# Unified through VisualizationEngine
engine = VisualizationEngine(db_config)
rpn_chart = engine.create_chart('dominance_plot', 'rpn', data)
binning_chart = engine.create_chart('price_color', 'binning', data)
```

## 📊 Available Visualizations

### **RPN Features**
- RPN & Market Dominance Analysis
- CWT Detrending Analysis  
- Kalman Filtering Analysis
- Correlation Analysis
- Extreme Conditions Detection

### **Binning Features**
- Price Colored by Bins
- Bin Analysis Dashboard
- Clustering Analysis
- Momentum Patterns
- Market Regime Analysis

### **Dominance Features**
- Dominance Status Overview
- Market Regime Dashboard
- Dominance Patterns Analysis
- Dominance Correlation Matrix

### **Signal Features**
- Trading Signals Overview
- Liquidations Analysis
- Trading Signals Dashboard
- Ceiling/Bottom Analysis
- Reversal Signals Analysis
- Signal Performance Metrics

### **Statistical Analysis**
- Statistical Overview Dashboard
- Distribution Analysis
- Feature Correlations
- Risk Metrics Analysis

## 🔧 Configuration

### **Command Line Options**
```bash
# Server options
--host 127.0.0.1          # Host to bind to
--port 8050               # Port to run on
--debug                   # Enable debug mode

# Database options
--db-host localhost       # Database host
--db-port 5432           # Database port
--db-name btc_features   # Database name
--db-user username       # Database user
--db-password password   # Database password
--db-config config.json  # Database config file

# Logging options
--log-level INFO         # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### **Database Config File**
```json
{
    "host": "localhost",
    "port": 5432,
    "database": "btc_features", 
    "user": "your_user",
    "password": "your_password"
}
```

## 🔌 API Integration

The dashboard includes a RESTful API for programmatic access:

```bash
# Get feature data
curl "http://localhost:8050/api/features?hours=24"

# Get specific plot data
curl "http://localhost:8050/api/plots/price_color?hours=168"

# Get latest signals
curl "http://localhost:8050/api/latest_signals?hours=24"

# Get performance metrics
curl "http://localhost:8050/api/performance?hours=168"

# Get database statistics
curl "http://localhost:8050/api/stats"
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/WTRCodeHub/WTRFHMVGg"
```

#### **Database Connection Issues**
```bash
# Test database connection
python -c "from feature_engineer.db.database_manager import DatabaseManager; db = DatabaseManager(); print('Database connection successful')"

# Check database tables
python -c "from feature_engineer.data_reader.reader import BTCDataReader; reader = BTCDataReader(); print(reader.get_feature_stats('btc_features_rpn'))"
```

#### **Port Already in Use**
```bash
# Use different port
python run_dashboard.py --port 8051

# Or kill existing process
lsof -ti:8050 | xargs kill -9
```

#### **Missing Data**
```bash
# Run feature engineering pipeline first
cd feature_engineer
python main.py --full-pipeline

# Check data availability
python main.py --status
```

## 🚀 Development

### **Adding New Features**

1. **Create Visualizer Class**
```python
# In feature_engineer/advanced_visualization/
class NewFeatureVisualizer:
    def create_new_chart(self) -> go.Figure:
        # Implementation
        pass
```

2. **Add to VisualizationEngine**
```python
# In visualization_engine.py
def _get_chart_methods(self, category: str):
    method_maps = {
        'new_category': {
            'new_chart': 'create_new_chart'
        }
    }
```

3. **Create Feature Page**
```python
# In feature_pages.py
class NewFeaturePage(BasePage):
    category = 'new_category'
    # Implementation
```

4. **Add to Main Dashboard**
```python
# In main_dashboard.py
self.pages = {
    'new_category': NewFeaturePage(self.engine)
}
```

### **Customizing Styling**

The dashboard uses a dark financial theme that can be customized in `main_dashboard.py`:

```css
/* Custom CSS in app.index_string */
body {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
}

.chart-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 32px rgba(0, 212, 255, 0.2);
}
```

## 📈 Performance

- **Caching**: Visualizers are cached to avoid recreation
- **Lazy Loading**: Charts are generated on-demand
- **Batch Operations**: Multiple charts can be generated in parallel
- **Responsive Updates**: Only necessary components are updated
- **Memory Management**: Automatic cleanup of unused visualizers

## 🔒 Security

- **Input Validation**: All user inputs are validated
- **SQL Injection Protection**: Parameterized queries
- **XSS Prevention**: Dash built-in protections
- **CORS Configuration**: Configurable cross-origin settings

## 📚 Technical Details

### **Chart Generation Flow**
```python
1. User selects time range → ComponentCallbacks.parse_time_range()
2. Data retrieval → VisualizationEngine.get_data()
3. Visualizer creation → VisualizationEngine.get_visualizer()
4. Chart generation → VisualizationEngine.create_chart()
5. Display update → Dash callback system
```

### **Data Pipeline**
```python
Database → BTCDataReader → VisualizationEngine → FeaturePage → Dashboard
```

### **Error Handling**
- Graceful degradation with fallback charts
- Comprehensive error logging
- User-friendly error messages
- Automatic retry mechanisms

---

## 🎉 Summary

The unified dashboard successfully integrates all existing visualization functions into a single, professional website with:

✅ **Complete Integration**: All visualization modules unified  
✅ **Professional UI**: Dark financial theme with responsive design  
✅ **Feature-Specific Pages**: Dedicated pages for each category  
✅ **Real-time Updates**: Live data with auto-refresh  
✅ **API Integration**: RESTful API for programmatic access  
✅ **Easy Deployment**: Simple command-line launcher  
✅ **Extensible Architecture**: Easy to add new features  

**Ready to use!** 🚀