# FHMV Model Training Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Test (Recommended First Step)
```bash
python quick_test.py
```
This runs a minimal test to verify all components work correctly.

### 3. Full Training Pipeline
```bash
python train_fhmv_model.py
```

## What the Training Pipeline Does

### 🔄 Complete Workflow
1. **Synthetic Data Generation**: Creates 1000 hours of realistic market data with 6 regime types
2. **Model Training**: Trains FHMV with Student's t-Mixture emissions using EM algorithm
3. **Visualization**: Creates comprehensive plots showing training progress and results
4. **Model Persistence**: Saves trained model to `outputs/fhmv_model.pkl`
5. **Prediction Demo**: Shows how to make predictions on new data
6. **Incremental Training Plan**: Provides strategy for continuous model updates

### 📊 Generated Outputs
- `outputs/fhmv_training_results.png`: Training visualization dashboard
- `outputs/fhmv_model.pkl`: Saved trained model
- Console logs: Detailed training progress and metrics

### 🎯 Key Features Demonstrated

#### 1. Synthetic Data Generation
- **6 Market Regimes**: ST, VT, RC, RHA, HPEM, AMB
- **10 Features**: LM, SR, Abs, RoC, RoC², P, T, Vol, RPN, Delta
- **Realistic Dynamics**: Regime persistence, cross-correlations, noise

#### 2. FHMV Model Training
- **Factor-based Architecture**: Persistence, Jump, Leverage components
- **Student's t-Mixture Emissions**: Fat-tail modeling for extreme events
- **Robust Numerical Implementation**: Enhanced stability and error handling

#### 3. Model Evaluation
- **Regime Classification**: Confusion matrix and accuracy metrics
- **Training Convergence**: Log-likelihood progression monitoring
- **Parameter Learning**: FHMV factor estimation and DoF optimization

#### 4. Production-Ready Features
- **Model Serialization**: Complete save/load functionality
- **Feature Processing**: Standardization with proper train/test separation
- **Inference Pipeline**: Viterbi algorithm and state probability estimation

## Incremental Training Strategy

### 🔄 Continuous Learning Approach
```python
# Pseudocode for incremental training
def incremental_update(new_data):
    # 1. Check if retraining needed
    if should_retrain(new_data):
        # 2. Combine with historical data
        combined_data = prepare_training_data(new_data)
        
        # 3. Warm-start from previous model
        model.initialize_from_previous()
        
        # 4. Run reduced EM iterations
        model.train(combined_data, max_iter=10)
        
        # 5. Validate and deploy
        if validate_model(model):
            deploy_model(model)
```

### 📈 Monitoring Triggers
- **Data Volume**: Retrain every 100-500 new observations
- **Performance Degradation**: Monitor accuracy drops
- **Feature Drift**: Detect distribution changes
- **Regime Anomalies**: Unusual transition patterns

### ⚡ Optimization Tips
- Use warm-start initialization from previous model
- Reduce EM iterations for incremental updates (10-20 vs 50)
- Implement parallel processing for E-step calculations
- Cache transition matrices when possible

## Model Architecture Details

### 🏗️ FHMV Structure
```
Combined States = Persistence × Jump × Leverage
                = 2^2 × 2 × 2 = 8 states

Mapped to 6 Final Regimes:
- ST (Stable Trend)
- VT (Volatile Trend) 
- RC (Range-bound/Consolidation)
- RHA (Reversal/High Absorption)
- HPEM (High Pressure/Extreme Move)
- AMB (Ambiguous)
```

### 🎲 Emission Model
- **Student's t-Mixture**: K=2 components per state
- **Degrees of Freedom**: Adaptive estimation (2.01-100)
- **Effective Covariance**: FHMV factor scaling
- **Numerical Stability**: Log-space calculations, Cholesky decomposition

### 🔧 Training Algorithm
1. **Initialization**: K-means clustering + reasonable parameter bounds
2. **E-Step**: Forward-backward algorithm with log-space calculations
3. **M-Step**: ECM approach (StMM params → FHMV factors)
4. **Convergence**: Log-likelihood tolerance (1e-4) or max iterations (50)

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all modules in `fhmv/` package have `__init__.py`
2. **Memory Issues**: Reduce `n_samples` in config for large datasets
3. **Convergence Issues**: Check feature scaling and initial parameters
4. **Numerical Instability**: Monitor DoF estimation bounds

### Performance Tips
- Start with quick_test.py to verify basic functionality
- Use smaller datasets (100-500 samples) for initial development
- Monitor training convergence through log-likelihood plots
- Validate regime predictions against known market patterns

## Next Steps

1. **Real Data Integration**: Replace synthetic data with actual market data
2. **Feature Engineering**: Implement Delta EWMA smoothing and other preprocessing
3. **Signal Generation**: Integrate with CLAD and l3 signal modules
4. **Backtesting**: Implement comprehensive strategy evaluation
5. **Production Deployment**: Set up automated retraining pipelines

## Support

For issues or questions:
1. Check console logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Start with quick_test.py to isolate issues
4. Review training visualizations for convergence problems