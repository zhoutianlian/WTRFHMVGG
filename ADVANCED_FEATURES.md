# Advanced Features Integration Guide

This document provides comprehensive documentation for the advanced features that have been integrated into the BTC feature engineering pipeline.

## Overview

The advanced features system extends the existing feature engineering pipeline with sophisticated financial indicators and analysis tools. These features are designed to work with the existing RPN (Risk Priority Number), binning, dominance, and signal features.

## Architecture

### Feature Registry System

The advanced features are managed through a centralized registry system:

- **Registry Location**: `feature_engineer/advanced_features/registry.py`
- **Base Class**: `feature_engineer/advanced_features/base.py`
- **Feature Categories**: Volatility, Momentum, Statistical, Trend

### Integration Flow

```
Raw Data → Preprocessing → RPN Features → Bin Features → Dominance Features → Signal Features → **Advanced Features**
```

Advanced features are processed after all main pipeline features to ensure dependencies are met.

## Advanced Feature Categories

### 1. Volatility Features

#### Adaptive Volatility (`adaptive_volatility`)
- **Purpose**: Volatility that adapts to market regime changes
- **Method**: Combines short-term (6h) and long-term (24h) volatilities
- **Output**: `vol_adaptive`
- **Formula**: `weight * short_vol + (1 - weight) * long_vol`

#### GARCH Volatility (`garch_volatility`)  
- **Purpose**: GARCH-like volatility estimation with clustering effects
- **Parameters**: α=0.1, β=0.85, ω=0.05
- **Output**: `vol_garch`
- **Formula**: `σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)`

### 2. Trend Analysis Features

#### KAMA (Kaufman Adaptive Moving Average)
- **Features**: `kama_fll`, `kama_fsl`, `kama_price`
- **Purpose**: Adaptive moving average that adjusts to market efficiency
- **Application**: Applied to FLL, FSL liquidation data and price
- **Parameters**: Window=10-14 periods

#### Kalman Slope (`kalman_slope`)
- **Purpose**: Kalman filter-based trend slope estimation
- **Outputs**: 
  - `kalman_slope`: Main slope estimate
  - `kalman_slope_filtered_price`: Filtered price
  - `kalman_slope_innovation`: Prediction error
  - `kalman_slope_smooth`: Smoothed slope
  - `kalman_slope_direction`: Trend direction signals
  - `kalman_slope_acceleration`: Slope acceleration
  - `kalman_slope_volatility`: Slope volatility

### 3. Momentum Indicators

#### RSI & Stochastic Oscillator (`momentum_indicators`)
- **Outputs**:
  - `price_rsi`: Relative Strength Index (14-period)
  - `price_stoch_k`: Stochastic %K
  - `price_stoch_d`: Stochastic %D (smoothed)
- **Signals**: Overbought (>70) and oversold (<30) conditions

#### MACD (`macd_price`)
- **Outputs**:
  - `price_macd`: MACD line (12-26 EMA difference)
  - `price_macd_signal`: Signal line (9-period EMA of MACD)
  - `price_macd_histogram`: MACD - Signal

#### ROC Dynamics
- **Features**: `roc_dynamics_fll`, `roc_dynamics_price`
- **Outputs**:
  - `fll_velocity`, `price_velocity`: Rate of change (velocity)
  - `fll_acceleration`, `price_acceleration`: Change in velocity
- **Methods**: Gaussian and Savitzky-Golay smoothing

### 4. Statistical Analysis Features

#### Spike Ratio Analysis
- **Features**: `spike_ratio_fll`, `spike_ratio_fsl`, `spike_ratio_price`
- **Purpose**: Identify spikes and deviations from trend
- **Outputs** (example for FLL):
  - `fll_spike_kama`: Main spike ratio
  - `fll_spike_kama_strength`: Deviation magnitude
  - `fll_spike_kama_direction`: Spike direction (1=up, -1=down, 0=neutral)
  - `fll_spike_kama_zscore`: Z-score of spike ratio
  - `fll_spike_kama_extreme`: Extreme spike detection

#### Statistical Moments (`statistical_moments`)
- **Purpose**: Higher-order distribution analysis
- **Outputs**:
  - `price_skewness`: Distribution asymmetry
  - `price_skew_signal`: Skewness-based signals
  - `price_kurtosis`: Tail heaviness
  - `price_excess_kurtosis`: Excess kurtosis (kurtosis - 3)
  - `price_fat_tails`: Fat tail detection
  - `price_jarque_bera`: Normality test statistic
  - `price_non_normal`: Non-normality indicator
  - `price_cv`: Coefficient of variation

## Database Schema

### Advanced Features Table (`btc_features_advanced`)

The advanced features are stored in a dedicated table with the following structure:

```sql
CREATE TABLE btc_features_advanced (
    id SERIAL PRIMARY KEY,
    time TIMESTAMP NOT NULL UNIQUE,
    
    -- KAMA Features
    fll_kama DECIMAL(20, 8),
    fsl_kama DECIMAL(20, 8), 
    price_kama DECIMAL(20, 8),
    
    -- Spike Ratio Features (with all derived metrics)
    fll_spike_kama DECIMAL(10, 8),
    fll_spike_kama_strength DECIMAL(10, 8),
    fll_spike_kama_direction INTEGER,
    fll_spike_kama_zscore DECIMAL(10, 8),
    fll_spike_kama_extreme INTEGER,
    -- ... (similar for FSL and price)
    
    -- ROC Dynamics
    fll_velocity DECIMAL(20, 8),
    fll_acceleration DECIMAL(20, 8),
    price_velocity DECIMAL(20, 8),
    price_acceleration DECIMAL(20, 8),
    
    -- Momentum Indicators
    price_rsi DECIMAL(10, 8),
    price_stoch_k DECIMAL(10, 8),
    price_stoch_d DECIMAL(10, 8),
    
    -- MACD
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
    price_kurtosis DECIMAL(10, 8),
    price_jarque_bera DECIMAL(10, 8),
    price_non_normal INTEGER,
    
    -- Volatility Features
    vol_adaptive DECIMAL(10, 8),
    vol_garch DECIMAL(10, 8),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Visualization Components

### Advanced Dashboard Components

Location: `feature_engineer/advanced_visualization/`

#### 1. FinancialDashboard (`dashboard.py`)
- **Purpose**: Professional financial dashboard with interactive features
- **Key Methods**:
  - `create_main_dashboard()`: Multi-feature overview
  - `create_advanced_price_analysis()`: Price with advanced indicators
  - `create_correlation_heatmap()`: Feature correlation analysis

#### 2. DistributionAnalyzer (`interactive.py`)
- **Purpose**: Advanced distribution analysis with statistical testing
- **Key Methods**:
  - `create_distribution_dashboard()`: Multi-feature distributions
  - `create_qq_plot()`: Q-Q plots for normality testing

#### 3. CorrelationAnalyzer (`interactive.py`)
- **Purpose**: Enhanced correlation analysis with clustering
- **Key Methods**:
  - `create_enhanced_correlation_matrix()`: Clustered correlation matrix
  - `create_rolling_correlation_analysis()`: Time-varying correlations

#### 4. AdvancedCharts (`charts.py`)
- **Purpose**: Specialized technical analysis charts
- **Key Methods**:
  - `create_multi_timeframe_analysis()`: Multi-timeframe feature analysis
  - `create_momentum_analysis_chart()`: Comprehensive momentum dashboard
  - `create_regime_analysis_chart()`: Regime-based feature analysis

### Color Scheme and Styling

Professional dark theme optimized for financial analysis:

- **Primary Colors**: Cyan (#00D4FF), Purple (#7B61FF), Pink (#FF6B9D)
- **Status Colors**: Green (bullish), Red (bearish), Orange (neutral)
- **Background**: Dark (#0A0E27) with card backgrounds (#1A1F3A)

## Usage Examples

### Running Advanced Feature Engineering

```python
from feature_engineer.feature_engineering.feature_engineer import FeatureEngineer
from feature_engineer.db.database_manager import DatabaseManager

# Initialize
db = DatabaseManager()
engineer = FeatureEngineer(db)

# Run complete pipeline (includes advanced features)
results = engineer.run_feature_engineering()
print(f"Advanced features processed: {results.get('advanced', 0)} records")
```

### Accessing Advanced Features

```python
from feature_engineer.data_reader.reader import BTCDataReader

reader = BTCDataReader(db)
df = reader.get_all_features()

# Check available advanced features
advanced_cols = [col for col in df.columns if any(keyword in col.lower() 
                for keyword in ['kama', 'spike', 'roc', 'rsi', 'kalman', 'vol_adaptive'])]
print(f"Available advanced features: {advanced_cols}")
```

### Creating Advanced Visualizations

```python
from feature_engineer.advanced_visualization import FinancialDashboard

# Create dashboard
dashboard = FinancialDashboard(df)

# Generate main dashboard
fig = dashboard.create_main_dashboard(['price_kama', 'kalman_slope', 'price_rsi', 'vol_adaptive'])

# Create advanced price analysis
price_fig = dashboard.create_advanced_price_analysis()
```

### Using the Feature Registry

```python
from feature_engineer.advanced_features.registry import get_registry

registry = get_registry()

# List all available features
features = registry.list_features()
print(f"Available features: {features}")

# Get features by category
categories = registry.get_features_by_category()
print(f"Volatility features: {categories['volatility']}")

# Get feature information
info = registry.get_feature_info('kalman_slope')
print(f"Kalman slope info: {info}")
```

## Web Dashboard Integration

The advanced features are integrated into the web dashboard with a new "Advanced Features" tab:

### Dashboard Features
1. **Advanced Features Dashboard**: Multi-feature overview with professional styling
2. **Advanced Price Analysis**: Price with KAMA, Kalman slope, and RSI overlays
3. **Feature Correlation Analysis**: Enhanced correlation matrix with clustering
4. **Momentum Analysis**: Comprehensive momentum indicators dashboard
5. **Multi-Timeframe Analysis**: Feature analysis across different time horizons

### Access URL
- Local: `http://localhost:8050`
- Tab: "Advanced Features"

## Performance Considerations

### Execution Order
Features are processed in optimal dependency order:
1. Basic volatility and trend features (adaptive volatility, GARCH, Kalman slope)
2. KAMA features (depend on existing data)
3. Momentum indicators (RSI, MACD)
4. Spike ratios (may depend on KAMA)
5. ROC dynamics
6. Statistical features

### Memory Management
- Batch processing: 10,000 records per batch
- Feature validation: Dependencies checked before processing
- Error handling: Individual feature failures don't stop the pipeline

### Database Optimization
- Indexes on time columns
- Vacuum analyze after processing
- Conflict resolution with upserts

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```
   ERROR: Some features have missing dependencies: {'feature_name': ['missing_column']}
   ```
   **Solution**: Ensure main pipeline features are processed first

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'advanced_features'
   ```
   **Solution**: Check Python path includes feature_engineer directory

3. **Visualization Errors**
   ```
   Error loading advanced features: ...
   ```
   **Solution**: Verify advanced features data exists in database

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
1. **Machine Learning Features**: PCA, clustering-based features
2. **Cross-Asset Features**: Correlation with other cryptocurrencies
3. **Market Microstructure**: Order book analysis features
4. **Real-time Processing**: Streaming feature updates

### Customization Options
1. **Custom Features**: Extend BaseAdvancedFeature class
2. **Parameter Tuning**: Modify registry parameters
3. **Visualization Themes**: Customize color schemes and layouts

## References

- **KAMA**: Kaufman, Perry J. "Smarter Trading" (1995)
- **Kalman Filters**: Kalman, R.E. "A new approach to linear filtering" (1960)
- **GARCH Models**: Bollerslev, Tim "Generalized autoregressive conditional heteroskedasticity" (1986)
- **Technical Analysis**: Murphy, John J. "Technical Analysis of the Financial Markets" (1999)