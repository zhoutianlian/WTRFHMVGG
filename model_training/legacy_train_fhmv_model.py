#!/usr/bin/env python3
"""
FHMV Model Training Script with Comprehensive Workflow

This script demonstrates the complete FHMV model training workflow including:
1. Synthetic data generation based on FHMV model specifications
2. Model training with visualization
3. Model persistence (save/load)
4. Prediction and inference
5. Incremental training pipeline for new data

Author: FHMV Development Team
Date: 2024
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fhmv.core_engine.fhmv_core_engine import FHMVCoreEngine
from fhmv.feature_engineering.feature_engineering_module import FeatureEngineeringModule

class FHMVTrainingPipeline:
    """
    Comprehensive FHMV training pipeline with synthetic data generation,
    model training, visualization, and incremental learning capabilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the training pipeline with configuration."""
        self.config = config or self._get_default_config()
        self.feature_names = [
            'LM', 'SR', 'Abs', 'RoC', 'RoC2', 
            'P', 'T', 'Vol', 'RPN', 'Delta'
        ]
        self.regime_names = ['ST', 'VT', 'RC', 'RHA', 'HPEM', 'AMB']
        
        # Initialize components
        self.fhmv_engine = None
        self.feature_processor = None
        self.training_history = []
        
        print("🚀 FHMV Training Pipeline Initialized")
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for FHMV model."""
        return {
            # Model Architecture
            "num_fhmv_regimes": 6,
            "num_persistence_chains": 2,
            "num_jump_states": 2,
            "num_leverage_states": 2,
            "num_stmm_mix_components": 2,
            "num_features": 10,
            
            # Training Parameters
            "em_max_iterations": 50,
            "em_tolerance": 1e-4,
            "dof_min": 2.01,
            "dof_max": 100.0,
            
            # FHMV Parameters
            "c1_bounds": [1.001, 3.0],
            "theta_c_bounds": [0.1, 0.99],
            "m1_bounds": [1.001, 3.0],
            "theta_m_bounds": [0.1, 0.99],
            "l1_bounds": [1.001, 3.0],
            "theta_l_bounds": [0.1, 0.99],
            "sigma_sq_bounds": [1e-4, 5.0],
            
            # Data Generation
            "n_samples": 1000,
            "n_train": 800,
            "n_test": 200,
            
            # Combined to Final Regime Mapping (8 combined states -> 6 final regimes)
            "combined_to_final_regime_mapping": {
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0, 7: 1,
                8: 2, 9: 3, 10: 4, 11: 5, 12: 0, 13: 1, 14: 2, 15: 3
            }
        }
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic FHMV-compatible data with realistic market characteristics.
        
        Args:
            n_samples: Number of hourly samples to generate
            
        Returns:
            DataFrame with 10 features and realistic market dynamics
        """
        print(f"📊 Generating {n_samples} samples of synthetic market data...")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate time index
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Initialize feature arrays
        features = np.zeros((n_samples, 10))
        
        # Generate regime sequence (hidden states)
        regime_probs = np.array([0.3, 0.2, 0.25, 0.1, 0.1, 0.05])  # ST, VT, RC, RHA, HPEM, AMB
        regime_sequence = np.random.choice(6, size=n_samples, p=regime_probs)
        
        # Add persistence to regime sequence
        for i in range(1, n_samples):
            if np.random.random() < 0.85:  # 85% persistence
                regime_sequence[i] = regime_sequence[i-1]
        
        # Generate features based on regime characteristics
        for i, regime in enumerate(regime_sequence):
            if regime == 0:  # ST (Stable Trend)
                features[i] = self._generate_st_features(i, features)
            elif regime == 1:  # VT (Volatile Trend)
                features[i] = self._generate_vt_features(i, features)
            elif regime == 2:  # RC (Range-bound/Consolidation)
                features[i] = self._generate_rc_features(i, features)
            elif regime == 3:  # RHA (Reversal/High Absorption)
                features[i] = self._generate_rha_features(i, features)
            elif regime == 4:  # HPEM (High Pressure/Extreme Move)
                features[i] = self._generate_hpem_features(i, features)
            else:  # AMB (Ambiguous)
                features[i] = self._generate_amb_features(i, features)
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=self.feature_names, index=dates)
        
        # Add some realistic correlations and noise
        df = self._add_realistic_dynamics(df)
        
        # Store true regimes for evaluation
        df['true_regime'] = regime_sequence
        
        print(f"✅ Synthetic data generated: {df.shape}")
        print(f"📈 Regime distribution: {pd.Series(regime_sequence).value_counts().sort_index().to_dict()}")
        
        return df
    
    def _generate_st_features(self, i: int, features: np.ndarray) -> np.ndarray:
        """Generate features for Stable Trend regime."""
        feature_vec = np.zeros(10)
        
        # Low liquidation activity
        feature_vec[0] = np.random.exponential(0.5)  # LM
        feature_vec[1] = np.random.exponential(0.3)  # SR
        
        # Moderate absorption
        feature_vec[2] = np.random.normal(1.0, 0.2)  # Abs
        
        # Low rate of change
        feature_vec[3] = np.random.normal(0.0, 0.1)  # RoC
        feature_vec[4] = np.random.normal(0.0, 0.05)  # RoC2
        
        # Trending price with momentum
        prev_price = features[i-1, 5] if i > 0 else 100.0
        trend_strength = np.random.normal(0.02, 0.01)
        feature_vec[5] = prev_price * (1 + trend_strength)  # P
        feature_vec[6] = trend_strength * 10  # T (scaled trend)
        
        # Low volatility
        feature_vec[7] = np.random.gamma(2, 0.1)  # Vol
        
        # Slight directional bias in liquidations
        feature_vec[8] = np.random.normal(0.45 if trend_strength > 0 else 0.55, 0.05)  # RPN
        feature_vec[9] = np.random.normal(trend_strength * 100, 10)  # Delta
        
        return feature_vec
    
    def _generate_vt_features(self, i: int, features: np.ndarray) -> np.ndarray:
        """Generate features for Volatile Trend regime."""
        feature_vec = np.zeros(10)
        
        # Moderate to high liquidation activity
        feature_vec[0] = np.random.exponential(1.5)  # LM
        feature_vec[1] = np.random.exponential(1.0)  # SR
        
        # Decreasing absorption during moves
        feature_vec[2] = np.random.normal(0.7, 0.3)  # Abs
        
        # Higher rate of change
        feature_vec[3] = np.random.normal(0.0, 0.3)  # RoC
        feature_vec[4] = np.random.normal(0.0, 0.2)  # RoC2
        
        # Volatile price movements
        prev_price = features[i-1, 5] if i > 0 else 100.0
        trend_strength = np.random.normal(0.0, 0.03)
        feature_vec[5] = prev_price * (1 + trend_strength)  # P
        feature_vec[6] = trend_strength * 15  # T
        
        # High volatility
        feature_vec[7] = np.random.gamma(3, 0.2)  # Vol
        
        # Clear directional bias
        feature_vec[8] = np.random.normal(0.3 if trend_strength > 0 else 0.7, 0.1)  # RPN
        feature_vec[9] = np.random.normal(trend_strength * 150, 30)  # Delta
        
        return feature_vec
    
    def _generate_rc_features(self, i: int, features: np.ndarray) -> np.ndarray:
        """Generate features for Range-bound/Consolidation regime."""
        feature_vec = np.zeros(10)
        
        # Low to moderate liquidation
        feature_vec[0] = np.random.exponential(0.8)  # LM
        feature_vec[1] = np.random.exponential(0.6)  # SR
        
        # Fluctuating absorption
        feature_vec[2] = np.random.normal(1.2, 0.4)  # Abs
        
        # Oscillating around zero
        feature_vec[3] = np.random.normal(0.0, 0.15)  # RoC
        feature_vec[4] = np.random.normal(0.0, 0.1)  # RoC2
        
        # Range-bound price
        prev_price = features[i-1, 5] if i > 0 else 100.0
        range_center = 100.0
        range_width = 10.0
        feature_vec[5] = np.clip(
            prev_price + np.random.normal(0, 0.5),
            range_center - range_width, range_center + range_width
        )  # P
        feature_vec[6] = np.random.normal(0.0, 0.005)  # T (weak trend)
        
        # Moderate volatility
        feature_vec[7] = np.random.gamma(2.5, 0.15)  # Vol
        
        # Indecisive liquidations around 0.5
        feature_vec[8] = np.random.normal(0.5, 0.08)  # RPN
        feature_vec[9] = np.random.normal(0.0, 15)  # Delta
        
        return feature_vec
    
    def _generate_rha_features(self, i: int, features: np.ndarray) -> np.ndarray:
        """Generate features for Reversal/High Absorption regime."""
        feature_vec = np.zeros(10)
        
        # High liquidation (counter-trend)
        feature_vec[0] = np.random.exponential(2.0)  # LM
        feature_vec[1] = np.random.exponential(1.5)  # SR
        
        # Very high absorption (key differentiator)
        feature_vec[2] = np.random.normal(2.5, 0.5)  # Abs
        
        # Significant change in liquidation rate
        feature_vec[3] = np.random.normal(0.5, 0.3)  # RoC
        feature_vec[4] = np.random.normal(-0.2, 0.2)  # RoC2 (deceleration)
        
        # Reversing price
        prev_price = features[i-1, 5] if i > 0 else 100.0
        reversal_strength = np.random.normal(-0.02, 0.01)
        feature_vec[5] = prev_price * (1 + reversal_strength)  # P
        feature_vec[6] = reversal_strength * 12  # T
        
        # Expanding volatility
        feature_vec[7] = np.random.gamma(4, 0.2)  # Vol
        
        # Counter-trend liquidation dominance
        feature_vec[8] = np.random.normal(0.7, 0.1)  # RPN
        feature_vec[9] = np.random.normal(-50, 25)  # Delta (large counter-trend)
        
        return feature_vec
    
    def _generate_hpem_features(self, i: int, features: np.ndarray) -> np.ndarray:
        """Generate features for High Pressure/Extreme Move regime."""
        feature_vec = np.zeros(10)
        
        # Extreme liquidation (pro-trend)
        feature_vec[0] = np.random.exponential(3.0)  # LM
        feature_vec[1] = np.random.exponential(2.5)  # SR
        
        # Low absorption (key differentiator from RHA)
        feature_vec[2] = np.random.normal(0.3, 0.15)  # Abs
        
        # Extreme rate of change
        feature_vec[3] = np.random.normal(1.0, 0.4)  # RoC
        feature_vec[4] = np.random.normal(0.5, 0.3)  # RoC2 (acceleration)
        
        # Sharp accelerating move
        prev_price = features[i-1, 5] if i > 0 else 100.0
        extreme_move = np.random.normal(0.05, 0.02)
        feature_vec[5] = prev_price * (1 + extreme_move)  # P
        feature_vec[6] = extreme_move * 20  # T
        
        # Extreme volatility
        feature_vec[7] = np.random.gamma(5, 0.3)  # Vol
        
        # Extreme liquidation bias
        feature_vec[8] = np.random.normal(0.1 if extreme_move > 0 else 0.9, 0.05)  # RPN
        feature_vec[9] = np.random.normal(extreme_move * 200, 40)  # Delta
        
        return feature_vec
    
    def _generate_amb_features(self, i: int, features: np.ndarray) -> np.ndarray:
        """Generate features for Ambiguous regime."""
        feature_vec = np.zeros(10)
        
        # Conflicting signals
        feature_vec[0] = np.random.exponential(1.0)  # LM
        feature_vec[1] = np.random.exponential(0.8)  # SR
        feature_vec[2] = np.random.normal(1.0, 0.5)  # Abs
        
        # Low, fluctuating rates
        feature_vec[3] = np.random.normal(0.0, 0.2)  # RoC
        feature_vec[4] = np.random.normal(0.0, 0.15)  # RoC2
        
        # Indecisive price action
        prev_price = features[i-1, 5] if i > 0 else 100.0
        feature_vec[5] = prev_price * (1 + np.random.normal(0, 0.02))  # P
        feature_vec[6] = np.random.normal(0.0, 0.01)  # T
        
        # Moderate volatility
        feature_vec[7] = np.random.gamma(2.5, 0.15)  # Vol
        
        # Indecisive liquidations
        feature_vec[8] = np.random.normal(0.5, 0.1)  # RPN
        feature_vec[9] = np.random.normal(0.0, 20)  # Delta
        
        return feature_vec
    
    def _add_realistic_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic market dynamics and correlations."""
        # Add some autocorrelation
        for col in ['P', 'Vol', 'RPN']:
            for i in range(1, len(df)):
                df.iloc[i, df.columns.get_loc(col)] += 0.1 * df.iloc[i-1, df.columns.get_loc(col)]
        
        # Add cross-correlations
        df['Abs'] = df['Abs'] - 0.3 * df['Vol']  # Higher vol -> lower absorption
        df['RPN'] = np.clip(df['RPN'], 0.01, 0.99)  # Keep RPN in valid range
        
        # Add some noise
        for col in self.feature_names:
            df[col] += np.random.normal(0, 0.01, len(df))
        
        return df
    
    def train_model(self, train_data: pd.DataFrame, visualize: bool = True) -> Dict:
        """
        Train the FHMV model with comprehensive monitoring and visualization.
        
        Args:
            train_data: Training DataFrame with features
            visualize: Whether to create training visualizations
            
        Returns:
            Dictionary with training results and metrics
        """
        print("\n🔧 Starting FHMV Model Training...")
        
        # Prepare data
        feature_data = train_data[self.feature_names].copy()
        print(f"📊 Training data shape: {feature_data.shape}")
        
        # Initialize feature processor
        self.feature_processor = FeatureEngineeringModule(self.config)
        
        # Fit and transform features
        print("🔄 Processing features...")
        scaled_features = self.feature_processor.fit_transform(feature_data)
        
        # Initialize FHMV engine
        self.fhmv_engine = FHMVCoreEngine(self.config)
        
        # Train model with monitoring
        training_start = datetime.now()
        print(f"⏳ Training started at: {training_start}")
        
        try:
            # Custom training with monitoring
            self._train_with_monitoring(scaled_features)
            
            training_end = datetime.now()
            training_duration = training_end - training_start
            
            print(f"✅ Training completed in: {training_duration}")
            
            # Generate training results
            results = self._evaluate_training_results(scaled_features, train_data)
            
            if visualize:
                self._visualize_training_results(results, train_data)
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise e
    
    def _train_with_monitoring(self, scaled_features: pd.DataFrame):
        """Train model with detailed monitoring and logging."""
        if not isinstance(scaled_features, pd.DataFrame) or scaled_features.empty:
            raise ValueError("scaled_features must be non-empty DataFrame.")
        if scaled_features.shape[1] != self.fhmv_engine.num_features:
            raise ValueError(f"scaled_features must have {self.fhmv_engine.num_features} columns.")
        
        print("Starting FHMV Training Sub-Module...")
        self.fhmv_engine._initialize_all_parameters(scaled_features)

        previous_log_likelihood = -np.inf 
        self.training_history = []
        
        print(f"Beginning EM algorithm for up to {self.fhmv_engine.em_max_iterations} iterations...")
        
        converged = False
        for iteration in range(self.fhmv_engine.em_max_iterations):
            iteration_start = datetime.now()
            print(f"\n--- EM Iteration: {iteration + 1}/{self.fhmv_engine.em_max_iterations} ---")
            
            # E-Step
            print("Performing E-Step...")
            try: 
                e_step_outputs = self.fhmv_engine._e_step(scaled_features)
            except Exception as e: 
                print(f"Error in E-step: {e}")
                break
            
            current_log_likelihood = e_step_outputs['current_log_likelihood']
            print(f"Log-Likelihood: {current_log_likelihood:.6f}")
            
            if np.isnan(current_log_likelihood) or np.isinf(current_log_likelihood): 
                print("LL is NaN/Inf. Stopping.")
                break
            
            # M-Step
            print("Performing M-Step...")
            try: 
                self.fhmv_engine._m_step(scaled_features, e_step_outputs)
            except Exception as e: 
                print(f"Error in M-step: {e}")
                break
            
            iteration_end = datetime.now()
            iteration_duration = iteration_end - iteration_start
            
            # Store training history
            self.training_history.append({
                'iteration': iteration + 1,
                'log_likelihood': current_log_likelihood,
                'duration': iteration_duration.total_seconds(),
                'improvement': current_log_likelihood - previous_log_likelihood if iteration > 0 else 0
            })
            
            # Convergence check
            if iteration > 0 and abs(current_log_likelihood - previous_log_likelihood) < self.fhmv_engine.em_tolerance: 
                print(f"EM Algorithm Converged after {iteration + 1} iterations.")
                converged = True
                break
                
            if iteration > 0 and current_log_likelihood < previous_log_likelihood - abs(previous_log_likelihood * 1e-5): 
                print(f"Warning: LL decreased from {previous_log_likelihood:.4f} to {current_log_likelihood:.4f}.")
                
            previous_log_likelihood = current_log_likelihood
        
        if not converged and iteration == self.fhmv_engine.em_max_iterations - 1: 
            print(f"EM reached max iterations ({self.fhmv_engine.em_max_iterations}) without convergence.")
        
        print("\nFHMV Training Finished.")
    
    def _evaluate_training_results(self, scaled_features: pd.DataFrame, original_data: pd.DataFrame) -> Dict:
        """Evaluate training results and generate metrics."""
        print("\n📊 Evaluating training results...")
        
        # Get state probabilities and most likely sequence
        state_probs = self.fhmv_engine.get_state_probabilities(scaled_features, smoothed=True)
        predicted_regimes = self.fhmv_engine.get_most_likely_state_sequence(scaled_features)
        
        # Calculate final log-likelihood
        final_ll = self.training_history[-1]['log_likelihood'] if self.training_history else -np.inf
        
        # Model parameters summary
        fhmv_params = self.fhmv_engine.fhmv_params_
        
        results = {
            'final_log_likelihood': final_ll,
            'num_iterations': len(self.training_history),
            'converged': len(self.training_history) < self.config['em_max_iterations'],
            'training_history': self.training_history,
            'predicted_regimes': predicted_regimes,
            'state_probabilities': state_probs,
            'fhmv_parameters': fhmv_params,
            'feature_scaling_params': {
                'means': self.feature_processor.feature_means_.to_dict(),
                'stds': self.feature_processor.feature_stds_.to_dict()
            }
        }
        
        # If we have true regimes, calculate accuracy
        if 'true_regime' in original_data.columns:
            true_regimes = original_data['true_regime'].values
            accuracy = np.mean(predicted_regimes == true_regimes)
            results['regime_accuracy'] = accuracy
            print(f"📈 Regime prediction accuracy: {accuracy:.3f}")
        
        print(f"🎯 Final log-likelihood: {final_ll:.3f}")
        print(f"🔄 Iterations completed: {len(self.training_history)}")
        
        return results
    
    def _visualize_training_results(self, results: Dict, data: pd.DataFrame):
        """Create comprehensive visualizations of training results."""
        print("\n📈 Creating training visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FHMV Model Training Results', fontsize=16, fontweight='bold')
        
        # 1. Training Progress (Log-Likelihood)
        ax1 = axes[0, 0]
        if self.training_history:
            iterations = [h['iteration'] for h in self.training_history]
            log_likelihoods = [h['log_likelihood'] for h in self.training_history]
            ax1.plot(iterations, log_likelihoods, 'b-', marker='o', linewidth=2)
            ax1.set_xlabel('EM Iteration')
            ax1.set_ylabel('Log-Likelihood')
            ax1.set_title('Training Convergence')
            ax1.grid(True, alpha=0.3)
        
        # 2. Predicted vs True Regimes (if available)
        ax2 = axes[0, 1]
        if 'true_regime' in data.columns:
            true_regimes = data['true_regime'].values[:len(results['predicted_regimes'])]
            predicted_regimes = results['predicted_regimes']
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_regimes, predicted_regimes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                       xticklabels=self.regime_names, yticklabels=self.regime_names)
            ax2.set_title('Regime Classification Matrix')
            ax2.set_xlabel('Predicted Regime')
            ax2.set_ylabel('True Regime')
        
        # 3. Regime Sequence Visualization
        ax3 = axes[0, 2]
        time_idx = range(min(200, len(results['predicted_regimes'])))  # Show first 200 points
        ax3.plot(time_idx, results['predicted_regimes'][:len(time_idx)], 'r-', alpha=0.7, label='Predicted')
        if 'true_regime' in data.columns:
            ax3.plot(time_idx, data['true_regime'].values[:len(time_idx)], 'g--', alpha=0.7, label='True')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Regime')
        ax3.set_title('Regime Sequence (First 200 steps)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. State Probabilities Heatmap
        ax4 = axes[1, 0]
        prob_sample = results['state_probabilities'][:min(100, len(results['state_probabilities']))]
        sns.heatmap(prob_sample.T, cmap='viridis', ax=ax4, cbar_kws={'label': 'Probability'})
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Regime')
        ax4.set_title('State Probabilities (First 100 steps)')
        
        # 5. Feature Distributions by Regime
        ax5 = axes[1, 1]
        regime_counts = pd.Series(results['predicted_regimes']).value_counts().sort_index()
        ax5.bar(range(len(regime_counts)), regime_counts.values, color='skyblue', alpha=0.7)
        ax5.set_xlabel('Regime')
        ax5.set_ylabel('Count')
        ax5.set_title('Predicted Regime Distribution')
        ax5.set_xticks(range(len(self.regime_names)))
        ax5.set_xticklabels(self.regime_names, rotation=45)
        
        # 6. FHMV Parameters
        ax6 = axes[1, 2]
        params = results['fhmv_parameters']
        param_names = []
        param_values = []
        
        # Extract key parameters for visualization
        if 'c1_persistence' in params:
            param_names.extend(['c1_pers', 'θc_pers', 'm1_jump', 'θm_jump', 'l1_lev', 'σ²'])
            param_values.extend([
                params['c1_persistence'], params['theta_c_persistence'],
                params['m1_jump'], params['theta_m_jump'], 
                params['l1_leverage'], params['sigma_sq_overall']
            ])
        
        if param_names:
            ax6.bar(range(len(param_names)), param_values, color='lightcoral', alpha=0.7)
            ax6.set_xlabel('Parameter')
            ax6.set_ylabel('Value')
            ax6.set_title('FHMV Model Parameters')
            ax6.set_xticks(range(len(param_names)))
            ax6.set_xticklabels(param_names, rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/fhmv_training_results.png', dpi=300, bbox_inches='tight')
        print("📊 Training visualizations saved to 'outputs/fhmv_training_results.png'")
        
        plt.show()
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained FHMV model and preprocessing parameters.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Success status
        """
        if self.fhmv_engine is None or self.feature_processor is None:
            print("❌ No trained model to save.")
            return False
        
        try:
            model_state = {
                'fhmv_engine_state': {
                    'config': self.config,
                    'fhmv_params_': self.fhmv_engine.fhmv_params_,
                    'stmm_emission_params_': self.fhmv_engine.stmm_emission_params_,
                    'initial_state_probs_': self.fhmv_engine.initial_state_probs_
                },
                'feature_processor_state': {
                    'feature_means_': self.feature_processor.feature_means_,
                    'feature_stds_': self.feature_processor.feature_stds_,
                    'fitted_columns_': self.feature_processor.fitted_columns_
                },
                'training_history': self.training_history,
                'feature_names': self.feature_names,
                'regime_names': self.regime_names,
                'save_timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            print(f"✅ Model saved successfully to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a previously trained FHMV model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Success status
        """
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore FHMV engine
            self.config = model_state['fhmv_engine_state']['config']
            self.fhmv_engine = FHMVCoreEngine(self.config)
            self.fhmv_engine.fhmv_params_ = model_state['fhmv_engine_state']['fhmv_params_']
            self.fhmv_engine.stmm_emission_params_ = model_state['fhmv_engine_state']['stmm_emission_params_']
            self.fhmv_engine.initial_state_probs_ = model_state['fhmv_engine_state']['initial_state_probs_']
            
            # Restore feature processor
            self.feature_processor = FeatureEngineeringModule(self.config)
            self.feature_processor.feature_means_ = model_state['feature_processor_state']['feature_means_']
            self.feature_processor.feature_stds_ = model_state['feature_processor_state']['feature_stds_']
            self.feature_processor.fitted_columns_ = model_state['feature_processor_state']['fitted_columns_']
            
            # Restore other attributes
            self.training_history = model_state.get('training_history', [])
            self.feature_names = model_state.get('feature_names', self.feature_names)
            self.regime_names = model_state.get('regime_names', self.regime_names)
            
            save_time = model_state.get('save_timestamp', 'Unknown')
            print(f"✅ Model loaded successfully from: {filepath}")
            print(f"📅 Model saved at: {save_time}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            return False
    
    def predict(self, new_data: pd.DataFrame, return_probabilities: bool = False) -> Dict:
        """
        Make predictions on new data using the trained model.
        
        Args:
            new_data: DataFrame with same features as training data
            return_probabilities: Whether to return state probabilities
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        if self.fhmv_engine is None or self.feature_processor is None:
            raise RuntimeError("Model must be trained or loaded before making predictions.")
        
        print(f"🔮 Making predictions on {len(new_data)} samples...")
        
        # Prepare features
        feature_data = new_data[self.feature_names].copy()
        scaled_features = self.feature_processor.transform(feature_data)
        
        # Get predictions
        predicted_regimes = self.fhmv_engine.get_most_likely_state_sequence(scaled_features)
        
        results = {
            'predicted_regimes': predicted_regimes,
            'regime_names': [self.regime_names[r] for r in predicted_regimes]
        }
        
        if return_probabilities:
            state_probabilities = self.fhmv_engine.get_state_probabilities(scaled_features, smoothed=True)
            results['state_probabilities'] = state_probabilities
        
        print(f"✅ Predictions completed.")
        print(f"📊 Regime distribution: {pd.Series(predicted_regimes).value_counts().sort_index().to_dict()}")
        
        return results


def main():
    """Main execution function demonstrating the complete FHMV workflow."""
    print("🚀 FHMV Model Training & Evaluation Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FHMVTrainingPipeline()
    
    # 1. Generate synthetic data
    print("\n1️⃣ GENERATING SYNTHETIC DATA")
    print("-" * 30)
    data = pipeline.generate_synthetic_data(n_samples=1000)
    
    # Split data
    n_train = pipeline.config['n_train']
    train_data = data.iloc[:n_train].copy()
    test_data = data.iloc[n_train:].copy()
    
    print(f"📊 Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # 2. Train model
    print("\n2️⃣ TRAINING FHMV MODEL")
    print("-" * 30)
    try:
        results = pipeline.train_model(train_data, visualize=True)
        print(f"✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 3. Save model
    print("\n3️⃣ SAVING TRAINED MODEL")
    print("-" * 30)
    model_path = "outputs/fhmv_model.pkl"
    pipeline.save_model(model_path)
    
    # 4. Test model loading
    print("\n4️⃣ TESTING MODEL RELOAD")
    print("-" * 30)
    new_pipeline = FHMVTrainingPipeline()
    if new_pipeline.load_model(model_path):
        print("✅ Model reloaded successfully!")
    
    # 5. Make predictions on test data
    print("\n5️⃣ MAKING PREDICTIONS")
    print("-" * 30)
    predictions = new_pipeline.predict(test_data, return_probabilities=True)
    
    # Evaluate predictions if true regimes available
    if 'true_regime' in test_data.columns:
        true_regimes = test_data['true_regime'].values
        predicted_regimes = predictions['predicted_regimes']
        test_accuracy = np.mean(predicted_regimes == true_regimes)
        print(f"🎯 Test accuracy: {test_accuracy:.3f}")
    
    # 6. Incremental training plan
    print("\n6️⃣ INCREMENTAL TRAINING PLAN")
    print("-" * 30)
    incremental_plan = create_incremental_training_plan()
    print(incremental_plan)
    
    print("\n🎉 FHMV Pipeline Demo Completed Successfully!")


def create_incremental_training_plan() -> str:
    """Create a detailed plan for incremental model training."""
    plan = """
📋 INCREMENTAL TRAINING PLAN FOR FHMV MODEL

🔄 1. CONTINUOUS LEARNING STRATEGY
   • Collect new hourly market data continuously
   • Maintain a rolling window of recent data (e.g., 2000-5000 hours)
   • Trigger retraining based on:
     - Data volume: Every 100-500 new observations
     - Performance degradation: Monitor prediction accuracy
     - Market regime changes: Detect distribution shifts

🛠️ 2. IMPLEMENTATION APPROACH
   
   A) WARM-START TRAINING:
      • Initialize new model with previous parameters
      • Use previous StMM emissions as starting points
      • Reduce EM iterations (10-20 instead of 50)
   
   B) FEATURE DRIFT MONITORING:
      • Track feature distribution changes using KL-divergence
      • Update scaling parameters when drift detected
      • Implement feature importance tracking
   
   C) REGIME STABILITY MONITORING:
      • Monitor regime transition frequencies
      • Detect unusual regime patterns
      • Implement regime confidence scoring

📊 3. AUTOMATED PIPELINE COMPONENTS

   ```python
   class IncrementalFHMVTrainer:
       def __init__(self, base_model_path):
           self.base_pipeline = FHMVTrainingPipeline()
           self.base_pipeline.load_model(base_model_path)
           self.performance_history = []
           
       def should_retrain(self, new_data, current_predictions):
           # Check multiple criteria
           accuracy_degraded = self.check_accuracy_degradation(...)
           feature_drift = self.detect_feature_drift(new_data)
           regime_anomaly = self.detect_regime_anomalies(...)
           
           return accuracy_degraded or feature_drift or regime_anomaly
           
       def incremental_train(self, new_data, warm_start=True):
           # Combine with historical data
           combined_data = self.prepare_incremental_data(new_data)
           
           # Warm start from previous parameters
           if warm_start:
               self.initialize_warm_start()
           
           # Train with reduced iterations
           return self.base_pipeline.train_model(combined_data)
   ```

🔍 4. MONITORING METRICS
   • Model log-likelihood trends
   • Regime prediction accuracy
   • Feature scaling parameter stability
   • Computational performance metrics

⚡ 5. OPTIMIZATION STRATEGIES
   • Use parallel processing for E-step calculations
   • Implement batch processing for large datasets
   • Cache transition matrices when possible
   • Use approximate inference for real-time applications

📈 6. PRODUCTION DEPLOYMENT
   • Set up automated data pipelines
   • Implement A/B testing for model versions
   • Create alerting for model performance issues
   • Maintain model versioning and rollback capabilities

💾 7. DATA MANAGEMENT
   • Archive old model versions
   • Maintain feature engineering consistency
   • Implement data quality checks
   • Store training metadata and configurations

🔄 8. SCHEDULE RECOMMENDATIONS
   • Daily: Performance monitoring
   • Weekly: Feature drift assessment  
   • Monthly: Full model retraining
   • Quarterly: Architecture review and optimization
"""
    return plan


if __name__ == "__main__":
    main()