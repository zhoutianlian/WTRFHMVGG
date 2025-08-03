# FHMV Model Training Framework

## 🎯 Overview

Comprehensive training framework for FHMV models with configuration management, hyperparameter optimization, and monitoring capabilities.

## 📁 Directory Structure

```
model_training/
├── fhmv_config.yaml           # Comprehensive hyperparameter configuration
├── training_pipeline.py       # Core training pipeline class
├── run_training.py            # Simple training runner script
├── README.md                  # This documentation
└── outputs/                   # Training outputs and results
    ├── training_results.png   # Training visualization dashboard
    ├── training_config.yaml   # Used configuration
    └── training_results.yaml  # Training metrics and results
```

## 🚀 Quick Start

### **1. Basic Training**
```bash
# Train with default configuration
python model_training/run_training.py

# Train with custom config
python model_training/run_training.py --config model_training/fhmv_config.yaml

# Generate specific number of samples
python model_training/run_training.py --samples 2000
```

### **2. Advanced Training**
```bash
# Train with your own data
python model_training/run_training.py --data your_data.csv

# Run hyperparameter optimization
python model_training/run_training.py --hyperopt

# Skip visualizations (faster)
python model_training/run_training.py --no-visualize
```

### **3. Programmatic Usage**
```python
from model_training.training_pipeline import FHMVTrainingPipeline

# Initialize with config
pipeline = FHMVTrainingPipeline('model_training/fhmv_config.yaml')

# Generate or load data
data = pipeline.generate_synthetic_data(n_samples=1000)

# Train model
results = pipeline.train_model(data, visualize=True)

# Make predictions
predictions = pipeline.predict(test_data)
```

## ⚙️ Configuration System

### **Configuration File**: `fhmv_config.yaml`

The configuration file contains **150+ hyperparameters** organized into logical sections:

#### **🏗️ Model Architecture**
- `num_fhmv_regimes`: Number of final market regimes (3-10, recommended: 6)
- `num_features`: Input feature dimensions (8-15, recommended: 10)
- `num_mixture_components`: Mixture complexity (1-5, recommended: 3)
- `num_combined_states`: Intermediate states (8-20, recommended: 16)

#### **📊 Student's t-Distribution**
- `dof_bounds`: Degrees of freedom range ([2.1, 30.0] recommended)
- `initial_dof`: Starting DoF value (5.0 recommended)
- `dof_tolerance`: Convergence precision (1e-4 recommended)

#### **🔄 EM Algorithm**
- `em_max_iterations`: Training iteration limit (20-200, recommended: 50)
- `em_tolerance`: Convergence threshold (1e-6 to 1e-2, recommended: 1e-4)
- `num_random_starts`: Multiple initializations (1-10, recommended: 3)

#### **🔢 Numerical Stability**
- `min_eigenvalue`: Matrix stability (1e-8 to 1e-4, recommended: 1e-6)
- `regularization_strength`: Covariance regularization (1e-8 to 1e-4)
- `log_prob_clip`: Probability bounds ([-500, 500] recommended)

#### **📈 Factor Structure**
- `persistence_weight`: Regime persistence (0.0-1.0, recommended: 0.7)
- `jump_weight`: Transition speed (0.0-0.5, recommended: 0.2)
- `leverage_weight`: Asymmetric responses (0.0-0.3, recommended: 0.1)

#### **🛠️ Data Processing**
- `scaling_method`: Feature normalization ("standard", "minmax", "robust")
- `missing_value_method`: NaN handling ("forward_fill", "interpolate", "drop")
- `outlier_threshold`: Z-score cutoff (2.0-5.0, recommended: 3.0)

### **Key Hyperparameters to Adjust**

| Parameter | Impact | Recommended Range | Default |
|-----------|--------|------------------|---------|
| `num_fhmv_regimes` | Model complexity | 4-8 | 6 |
| `em_max_iterations` | Training quality | 30-100 | 50 |
| `dof_bounds.min` | Fat tail modeling | 2.0-3.0 | 2.1 |
| `persistence_weight` | Regime stability | 0.5-0.9 | 0.7 |
| `num_mixture_components` | Distribution flexibility | 2-4 | 3 |
| `em_tolerance` | Convergence precision | 1e-5 to 1e-3 | 1e-4 |

## 📊 Training Outputs

### **Visualization Dashboard**
The training creates a comprehensive 6-panel visualization:

1. **Training Convergence** - Log-likelihood improvement over iterations
2. **Regime Predictions** - Time series of predicted regimes
3. **Feature Distributions** - Feature separation by regime
4. **Model Configuration** - Summary of all parameters used
5. **Training Summary** - Key metrics and performance
6. **Transition Matrix** - Regime switching patterns

### **Output Files**
- `training_results.png` - Complete visualization dashboard
- `training_config.yaml` - Configuration used for training
- `training_results.yaml` - Numerical results and metrics

### **Training Metrics**
- **Log-likelihood**: Model fit quality (higher = better)
- **EM Iterations**: Convergence speed (fewer = faster convergence)
- **Training Time**: Computational efficiency
- **Convergence Status**: Whether training completed successfully

## 🎯 Performance Guidelines

### **Quality Thresholds**
- **🟢 Excellent**: Log-likelihood > -2.0, <30 iterations, <30 minutes
- **🟡 Good**: Log-likelihood > -3.0, <50 iterations, <60 minutes  
- **🟠 Acceptable**: Log-likelihood > -5.0, <100 iterations, <120 minutes
- **🔴 Poor**: Below acceptable thresholds - needs parameter tuning

### **Common Issues & Solutions**

#### **Low Log-likelihood (<-5.0)**
- Increase `num_mixture_components` (3→4)
- Adjust `dof_bounds` for your data characteristics
- Increase `em_max_iterations` (50→100)
- Check data quality and preprocessing

#### **Slow Convergence (>100 iterations)**
- Decrease `em_tolerance` (1e-4→1e-3)  
- Reduce `num_mixture_components` (4→3)
- Check for data issues (outliers, missing values)

#### **Training Instability**
- Increase `regularization_strength` (1e-6→1e-5)
- Adjust `min_eigenvalue` (1e-6→1e-5)
- Use more `num_random_starts` (3→5)

## 🔍 Hyperparameter Optimization

### **Manual Tuning Strategy**
1. **Start with defaults** - Use recommended values
2. **Adjust architecture** - Try different `num_fhmv_regimes` (4,6,8)
3. **Tune convergence** - Adjust `em_max_iterations` and `em_tolerance`
4. **Optimize distributions** - Modify `dof_bounds` and `num_mixture_components`
5. **Fine-tune stability** - Adjust regularization if needed

### **Grid Search Ranges**
```yaml
# Add to config file for systematic search
hyperparameter_search:
  search_ranges:
    num_mixture_components: [2, 5]
    em_max_iterations: [30, 100]
    dof_bounds_min: [2.0, 3.0]
    persistence_weight: [0.5, 0.9]
    regularization_strength: [1e-7, 1e-4]
```

## 💡 Best Practices

### **Data Preparation**
- Ensure consistent feature scaling
- Handle missing values appropriately  
- Remove extreme outliers (>3σ)
- Use sufficient training data (>500 samples)

### **Model Configuration**
- Start with 6 regimes for financial markets
- Use 3 mixture components for balance
- Set persistence_weight ≥ 0.6 for realistic regime duration
- Monitor training convergence carefully

### **Performance Optimization**
- Use multiple random starts for robustness
- Set reasonable iteration limits
- Enable early stopping
- Monitor memory usage for large datasets

### **Validation**
- Always visualize training results
- Check regime transition patterns
- Validate economic intuition of regimes
- Test on out-of-sample data

## 🚨 Troubleshooting

### **Common Error Messages**

#### **"Singular matrix" or "Cholesky decomposition failed"**
- Increase `regularization_strength`
- Check for perfect feature correlations
- Reduce `num_mixture_components`

#### **"EM algorithm did not converge"**
- Increase `em_max_iterations`
- Relax `em_tolerance`
- Try different random initialization

#### **"DoF estimation failed"**
- Adjust `dof_bounds` range
- Increase `max_dof_iterations`
- Check data for extreme outliers

### **Performance Issues**

#### **Very slow training**
- Reduce `num_mixture_components`
- Decrease `em_max_iterations`
- Use fewer features if possible
- Check for computational bottlenecks

#### **Poor regime separation**
- Increase `num_fhmv_regimes`
- Adjust `persistence_weight`
- Improve feature engineering
- Check data quality

## 📋 Dependencies

```python
# Core requirements
numpy>=1.21.0
pandas>=1.3.0  
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
pyyaml>=6.0
```

## 🎉 Quick Validation

After training, check these indicators:
- ✅ **Log-likelihood > -3.0**
- ✅ **Training converged < 50 iterations**  
- ✅ **Realistic regime durations (3-15 periods)**
- ✅ **Clear regime separation in visualizations**
- ✅ **Reasonable transition probabilities**

Your FHMV model is ready for evaluation and deployment when these criteria are met!