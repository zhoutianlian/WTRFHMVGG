# FHMV Unsupervised Evaluation Framework

## 🎯 Purpose

Evaluate FHMV model quality **without ground truth labels** using a streamlined, focused approach with BTC price visualization.

## 🚀 Quick Start

```bash
# Run evaluation
python evaluation_framework/run_evaluation.py
```

## 📊 What You Get

### **1. Comprehensive Quality Score (0-1)**
- **Prediction Confidence** (30%) - How certain the model is
- **Regime Quality** (25%) - How well regimes are separated  
- **Economic Value** (30%) - Trading strategy performance
- **Model Stability** (15%) - Regime persistence and transitions

### **2. BTC Price Visualization**
- **Color-coded regime predictions** over latest year
- **Confidence-based transparency** (higher confidence = more opaque)
- **6 distinct regime colors** for easy identification

### **3. Dashboard Components**
- BTC price with regime overlay
- Prediction confidence distribution
- Regime transition matrix
- Feature discrimination power
- Economic performance metrics
- Model health indicators

## 📈 Evaluation Metrics

### **Quality Thresholds**
- **🟢 Excellent (≥0.8)** - Production ready
- **🟡 Good (0.6-0.8)** - Minor improvements needed  
- **🟠 Acceptable (0.4-0.6)** - Significant improvements needed
- **🔴 Poor (<0.4)** - Major revisions required

### **Key Indicators**
- **Confidence >0.7** - Reliable predictions
- **Silhouette >0.3** - Good regime separation
- **Sharpe >0.5** - Economic value
- **Duration 3-15** - Realistic regime persistence

## 🎨 Regime Color Scheme

| Regime | Color | Market Condition |
|--------|-------|------------------|
| ST (Stable Trend) | 🔴 Red | Low volatility, consistent trend |
| VT (Volatile Trend) | 🔵 Blue | Medium volatility, trending |
| RC (Range-bound Consolidation) | 🟢 Green | Sideways market |
| RHA (Regime-Hybrid Adaptive) | 🟡 Yellow | Transition periods |
| HPEM (High-Persistence Extreme Move) | 🟠 Orange | Major market moves |
| AMB (Ambiguous) | 🟣 Purple | Uncertain conditions |

## 📁 Output Files

```
evaluation_framework/outputs/
└── unsupervised_evaluation_dashboard.png
    ├── BTC price with regime predictions
    ├── Confidence and quality metrics
    ├── Economic performance analysis
    └── Model health indicators
```

## 💡 Interpretation Guide

### **High Quality Model Should Have:**
- Mean confidence >0.7
- High confidence ratio >60%
- Silhouette score >0.3
- Positive Sharpe ratio
- Stable regime transitions
- Realistic regime durations

### **Red Flags:**
- Very low confidence (<0.5)
- Poor regime separation (silhouette <0.2)
- Negative economic performance
- Too frequent regime switches
- Very short/long regime durations

## 🔧 Customization

Edit `evaluation_framework/unsupervised_evaluator.py` to:
- Adjust regime color scheme
- Modify economic strategy logic
- Add custom metrics
- Change visualization layout

## 📋 Dependencies

- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn
- FHMV model pipeline

## 🎉 Benefits

✅ **No ground truth needed** - Works with real market data  
✅ **Economic validation** - Tests actual trading value  
✅ **Visual interpretation** - Easy-to-understand charts  
✅ **Comprehensive metrics** - Multiple evaluation dimensions  
✅ **Production ready** - Quality thresholds for deployment  
✅ **BTC focused** - Cryptocurrency market visualization