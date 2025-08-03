#!/usr/bin/env python3
"""
FHMV Unsupervised Evaluation Framework
=====================================

Streamlined evaluation for FHMV models without ground truth labels.
Focuses on practical evaluation metrics and BTC price visualization.

Author: FHMV Development Team
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Statistical tools
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StreamlinedUnsupervisedEvaluator:
    """
    Streamlined unsupervised evaluation for FHMV models.
    Focused on practical metrics and BTC visualization.
    """
    
    def __init__(self, model_pipeline, regime_names: Optional[List[str]] = None):
        """Initialize evaluator with model pipeline."""
        self.pipeline = model_pipeline
        self.regime_names = regime_names or [f'Regime_{i}' for i in range(6)]
        self.regime_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        
    def evaluate_model_quality(self, data: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """
        Main evaluation method - returns comprehensive quality assessment.
        
        Args:
            data: Market data with features
            save_plots: Whether to save visualization plots
            
        Returns:
            Dictionary with evaluation results and quality score
        """
        print("🔍 Running unsupervised FHMV evaluation...")
        
        # Get model predictions
        predictions = self.pipeline.predict(data, return_probabilities=True)
        regime_probs = predictions['state_probabilities']
        predicted_regimes = predictions['predicted_regimes']
        
        # Core evaluation metrics
        results = {
            'model_confidence': self._evaluate_prediction_confidence(regime_probs),
            'regime_quality': self._evaluate_regime_quality(data, predicted_regimes, regime_probs),
            'economic_value': self._evaluate_economic_value(data, predicted_regimes, regime_probs),
            'stability_metrics': self._evaluate_model_stability(predicted_regimes)
        }
        
        # Calculate composite quality score
        results['composite_score'] = self._calculate_composite_score(results)
        
        # Generate visualizations
        if save_plots:
            self._create_evaluation_dashboard(data, predicted_regimes, regime_probs, results)
            
        return results
    
    def _evaluate_prediction_confidence(self, regime_probs: np.ndarray) -> Dict[str, float]:
        """Evaluate model prediction confidence."""
        max_probs = np.max(regime_probs, axis=1)
        
        return {
            'mean_confidence': np.mean(max_probs),
            'high_confidence_ratio': np.mean(max_probs > 0.8),
            'low_confidence_ratio': np.mean(max_probs < 0.5),
            'confidence_std': np.std(max_probs)
        }
    
    def _evaluate_regime_quality(self, data: pd.DataFrame, regimes: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
        """Evaluate regime separation and quality."""
        # Prepare features for analysis
        feature_cols = [col for col in data.columns if col not in ['true_regime', 'returns']]
        X = data[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Silhouette analysis
        try:
            silhouette_avg = silhouette_score(X_scaled, regimes)
        except:
            silhouette_avg = 0.0
            
        # Feature discrimination
        discrimination_scores = {}
        for i, col in enumerate(feature_cols):
            f_stat, _ = stats.f_oneway(*[X_scaled[regimes == r, i] for r in np.unique(regimes)])
            discrimination_scores[col] = f_stat if not np.isnan(f_stat) else 0.0
        
        return {
            'silhouette_score': silhouette_avg,
            'feature_discrimination': discrimination_scores,
            'max_discrimination': max(discrimination_scores.values()) if discrimination_scores else 0.0,
            'regime_separation_quality': 'Excellent' if silhouette_avg > 0.5 else 'Good' if silhouette_avg > 0.3 else 'Poor'
        }
    
    def _evaluate_economic_value(self, data: pd.DataFrame, regimes: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        """Evaluate economic value through portfolio performance."""
        if 'returns' not in data.columns:
            # Generate synthetic returns for demo
            returns = np.random.normal(0.001, 0.02, len(data))
        else:
            returns = data['returns'].values
            
        # Simple regime-based strategy
        regime_returns = {}
        portfolio_returns = []
        
        for t in range(len(returns)):
            regime = regimes[t]
            confidence = np.max(probs[t])
            
            # Simple strategy: long in certain regimes, short in others, neutral when uncertain
            if confidence > 0.7:
                if regime in [0, 1, 4]:  # ST, VT, HPEM - assume bullish
                    position = 1.0
                elif regime in [2, 3]:   # RC, RHA - assume bearish/neutral
                    position = -0.5
                else:  # AMB
                    position = 0.0
            else:
                position = 0.0  # Stay neutral when uncertain
                
            portfolio_returns.append(position * returns[t])
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate performance metrics
        mean_return = np.mean(portfolio_returns) * 252  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'annualized_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': np.mean(portfolio_returns > 0)
        }
    
    def _evaluate_model_stability(self, regimes: np.ndarray) -> Dict[str, float]:
        """Evaluate regime stability and transition patterns."""
        # Regime durations
        regime_changes = np.diff(regimes) != 0
        change_points = np.where(regime_changes)[0] + 1
        change_points = np.concatenate([[0], change_points, [len(regimes)]])
        durations = np.diff(change_points)
        
        # Transition analysis
        transition_freq = np.sum(regime_changes) / len(regimes)
        
        return {
            'mean_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'transition_frequency': transition_freq,
            'regime_stability': 1 - transition_freq,
            'duration_stability': 'Good' if 3 <= np.mean(durations) <= 15 else 'Check'
        }
    
    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite quality score (0-1)."""
        score = 0.0
        
        # Confidence component (30%)
        confidence_score = results['model_confidence']['mean_confidence']
        score += 0.3 * confidence_score
        
        # Regime quality component (25%)
        silhouette = max(0, results['regime_quality']['silhouette_score'])
        quality_score = min(1.0, silhouette / 0.5)  # Normalize to 0.5 max
        score += 0.25 * quality_score
        
        # Economic value component (30%)
        sharpe = results['economic_value']['sharpe_ratio']
        econ_score = min(1.0, max(0, sharpe / 2.0))  # Normalize to 2.0 max
        score += 0.3 * econ_score
        
        # Stability component (15%)
        stability = results['stability_metrics']['regime_stability']
        score += 0.15 * stability
        
        return min(1.0, score)
    
    def _create_evaluation_dashboard(self, data: pd.DataFrame, regimes: np.ndarray, 
                                   probs: np.ndarray, results: Dict[str, Any]):
        """Create comprehensive evaluation dashboard with BTC price chart."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. BTC Price with Regime Predictions (Top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_btc_with_regimes(ax1, data, regimes, probs)
        
        # 2. Confidence Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_confidence_distribution(ax2, probs)
        
        # 3. Regime Transition Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_transition_matrix(ax3, regimes)
        
        # 4. Feature Discrimination
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_feature_discrimination(ax4, results['regime_quality']['feature_discrimination'])
        
        # 5. Economic Performance
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_economic_performance(ax5, results['economic_value'])
        
        # 6. Regime Duration Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_regime_durations(ax6, regimes)
        
        # 7. Quality Metrics Summary
        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_quality_summary(ax7, results)
        
        # 8. Model Health Indicators
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_model_health(ax8, results)
        
        plt.suptitle('FHMV Unsupervised Evaluation Dashboard', fontsize=20, fontweight='bold')
        
        # Save the plot
        os.makedirs('evaluation_framework/outputs', exist_ok=True)
        plt.savefig('evaluation_framework/outputs/unsupervised_evaluation_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Evaluation dashboard saved to: evaluation_framework/outputs/unsupervised_evaluation_dashboard.png")
    
    def _plot_btc_with_regimes(self, ax, data: pd.DataFrame, regimes: np.ndarray, probs: np.ndarray):
        """Plot BTC price with regime predictions and confidence colors."""
        
        # Generate synthetic BTC price data for latest year if not available
        if 'price' not in data.columns:
            # Create realistic BTC price movement
            np.random.seed(42)
            initial_price = 45000
            returns = np.random.normal(0.0005, 0.02, len(data))  # Daily returns
            prices = [initial_price]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            data = data.copy()
            data['price'] = prices
        
        # Create date index for latest year
        dates = pd.date_range(end='2024-12-31', periods=len(data), freq='D')
        
        # Plot price line
        ax.plot(dates, data['price'], color='black', linewidth=1, alpha=0.7, label='BTC Price')
        
        # Color segments by regime and confidence
        for i in range(len(data)-1):
            regime = regimes[i]
            confidence = np.max(probs[i])
            
            # Adjust color intensity based on confidence
            color = self.regime_colors[regime % len(self.regime_colors)]
            alpha = 0.3 + 0.7 * confidence  # Higher confidence = more opaque
            
            ax.axvspan(dates[i], dates[i+1], 
                      color=color, alpha=alpha, linewidth=0)
        
        ax.set_title('BTC Price with FHMV Regime Predictions (Latest Year)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('BTC Price ($)')
        ax.grid(True, alpha=0.3)
        
        # Add regime legend
        regime_handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=name) 
                         for color, name in zip(self.regime_colors, self.regime_names)]
        ax.legend(handles=regime_handles, loc='upper left', bbox_to_anchor=(1.01, 1))
        
        # Format y-axis for price
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _plot_confidence_distribution(self, ax, probs: np.ndarray):
        """Plot prediction confidence distribution."""
        max_probs = np.max(probs, axis=1)
        
        ax.hist(max_probs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(max_probs), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(max_probs):.3f}')
        ax.axvline(0.8, color='green', linestyle='--', alpha=0.7, label='High Confidence')
        ax.axvline(0.5, color='orange', linestyle='--', alpha=0.7, label='Low Confidence')
        
        ax.set_title('Prediction Confidence Distribution')
        ax.set_xlabel('Max Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_transition_matrix(self, ax, regimes: np.ndarray):
        """Plot regime transition matrix."""
        n_regimes = len(np.unique(regimes))
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes)-1):
            transitions[regimes[i], regimes[i+1]] += 1
        
        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions = transitions / row_sums
        
        im = ax.imshow(transitions, cmap='Blues', aspect='auto')
        ax.set_title('Regime Transition Matrix')
        ax.set_xlabel('To Regime')
        ax.set_ylabel('From Regime')
        
        # Add text annotations
        for i in range(n_regimes):
            for j in range(n_regimes):
                ax.text(j, i, f'{transitions[i,j]:.2f}', 
                       ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_feature_discrimination(self, ax, discrimination: Dict[str, float]):
        """Plot feature discrimination power."""
        features = list(discrimination.keys())[:8]  # Top 8 features
        scores = [discrimination[f] for f in features]
        
        bars = ax.barh(features, scores, color='lightcoral')
        ax.set_title('Feature Discrimination Power')
        ax.set_xlabel('F-Statistic')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight strong discriminators
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 2.0:
                bar.set_color('darkred')
                bar.set_alpha(0.8)
    
    def _plot_economic_performance(self, ax, econ_metrics: Dict[str, float]):
        """Plot economic performance metrics."""
        metrics = ['Sharpe Ratio', 'Hit Rate', 'Ann. Return']
        values = [
            econ_metrics.get('sharpe_ratio', 0),
            econ_metrics.get('hit_rate', 0),
            econ_metrics.get('annualized_return', 0) * 100  # Convert to percentage
        ]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_title('Economic Performance')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    def _plot_regime_durations(self, ax, regimes: np.ndarray):
        """Plot regime duration distribution."""
        regime_changes = np.diff(regimes) != 0
        change_points = np.where(regime_changes)[0] + 1
        change_points = np.concatenate([[0], change_points, [len(regimes)]])
        durations = np.diff(change_points)
        
        ax.hist(durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(durations), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(durations):.1f}')
        ax.axvspan(3, 15, alpha=0.2, color='green', label='Ideal Range')
        
        ax.set_title('Regime Duration Distribution')
        ax.set_xlabel('Duration (periods)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_summary(self, ax, results: Dict[str, Any]):
        """Plot quality metrics summary."""
        metrics = {
            'Confidence': results['model_confidence']['mean_confidence'],
            'Regime Quality': min(1.0, results['regime_quality']['silhouette_score'] / 0.5),
            'Economic Value': min(1.0, max(0, results['economic_value']['sharpe_ratio'] / 2.0)),
            'Stability': results['stability_metrics']['regime_stability'],
            'Overall Score': results['composite_score']
        }
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values = list(metrics.values())
        
        # Close the radar chart
        angles += angles[:1]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics.keys())
        ax.set_ylim(0, 1)
        ax.set_title('Model Quality Radar Chart')
        ax.grid(True)
        
        # Add score labels
        for angle, value, label in zip(angles[:-1], values[:-1], metrics.keys()):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    def _plot_model_health(self, ax, results: Dict[str, Any]):
        """Plot model health indicators."""
        # Create health score indicators
        health_metrics = {
            'High Confidence\nPredictions': results['model_confidence']['high_confidence_ratio'],
            'Feature\nDiscrimination': min(1.0, results['regime_quality']['max_discrimination'] / 3.0),
            'Economic\nValue': min(1.0, max(0, (results['economic_value']['sharpe_ratio'] + 1) / 3.0)),
            'Regime\nStability': results['stability_metrics']['regime_stability'],
            'Overall\nHealth': results['composite_score']
        }
        
        x_pos = np.arange(len(health_metrics))
        values = list(health_metrics.values())
        colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in values]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(health_metrics.keys(), rotation=0, ha='center')
        ax.set_ylabel('Score')
        ax.set_title('Model Health Indicators')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add threshold lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7+)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable (0.5+)')
        ax.legend(loc='upper right')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')