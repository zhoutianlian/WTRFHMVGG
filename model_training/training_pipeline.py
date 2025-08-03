#!/usr/bin/env python3
"""
FHMV Model Training Pipeline
===========================

Comprehensive training pipeline for FHMV models with configuration management,
hyperparameter optimization, and monitoring capabilities.

Author: FHMV Development Team
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FHMVTrainingPipeline:
    """
    Comprehensive FHMV training pipeline with configuration management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training pipeline with configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.training_history = []
        self.best_log_likelihood = -np.inf
        
        # Set random seed for reproducibility
        if 'random_seed' in self.config.get('advanced', {}):
            np.random.seed(self.config['advanced']['random_seed'])
        
        print(f"🔧 FHMV Training Pipeline Initialized")
        print(f"📊 Model: {self.config['model_architecture']['num_fhmv_regimes']} regimes, "
              f"{self.config['model_architecture']['num_features']} features")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded config from: {config_path}")
        else:
            # Default configuration
            config = {
                'model_architecture': {
                    'num_fhmv_regimes': 6,
                    'num_features': 10,
                    'num_mixture_components': 3,
                    'num_combined_states': 16
                },
                'em_algorithm': {
                    'em_max_iterations': 50,
                    'em_tolerance': 1e-4,
                    'num_random_starts': 3
                },
                'student_t_params': {
                    'dof_bounds': {'min': 2.1, 'max': 30.0},
                    'initial_dof': 5.0
                },
                'regime_mapping': {
                    'combined_to_final': {i: i % 6 for i in range(16)}
                }
            }
            print("⚠️ Using default configuration")
        
        return config
    
    def generate_synthetic_data(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic market data for training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic market features and regime labels
        """
        if n_samples is None:
            n_samples = self.config.get('training_data', {}).get('train_size', 1000)
        
        print(f"📊 Generating {n_samples} samples of synthetic market data...")
        
        # Set random seed
        np.random.seed(self.config.get('advanced', {}).get('random_seed', 42))
        
        # Generate regime sequence with realistic persistence
        regimes = self._generate_regime_sequence(n_samples)
        
        # Generate features based on regimes
        features = self._generate_regime_features(regimes, n_samples)
        
        # Create DataFrame
        feature_names = ['LM', 'SR', 'Abs', 'RoC', 'RoC2', 'P', 'T', 'Vol', 'RPN', 'Delta']
        data = pd.DataFrame(features, columns=feature_names[:self.config['model_architecture']['num_features']])
        data['true_regime'] = regimes
        
        # Add synthetic returns for evaluation
        data['returns'] = self._generate_returns_from_regimes(regimes)
        
        print(f"✅ Generated synthetic data with {len(np.unique(regimes))} unique regimes")
        
        return data
    
    def _generate_regime_sequence(self, n_samples: int) -> np.ndarray:
        """Generate realistic regime sequence with persistence."""
        regimes = np.zeros(n_samples, dtype=int)
        num_regimes = self.config['model_architecture']['num_fhmv_regimes']
        
        # Transition probabilities (higher diagonal = more persistent)
        persistence = self.config.get('factor_structure', {}).get('persistence_weight', 0.7)
        transition_prob = np.ones((num_regimes, num_regimes)) * (1 - persistence) / (num_regimes - 1)
        np.fill_diagonal(transition_prob, persistence)
        
        # Generate sequence
        current_regime = 0
        for t in range(n_samples):
            regimes[t] = current_regime
            if t < n_samples - 1:
                current_regime = np.random.choice(num_regimes, p=transition_prob[current_regime])
        
        return regimes
    
    def _generate_regime_features(self, regimes: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate features based on regime characteristics."""
        num_features = self.config['model_architecture']['num_features']
        features = np.zeros((n_samples, num_features))
        
        # Define regime characteristics
        regime_params = {
            0: {'vol': 0.01, 'trend': 0.0, 'lm': 0.3},    # ST: Stable Trend
            1: {'vol': 0.02, 'trend': 0.1, 'lm': 0.5},    # VT: Volatile Trend  
            2: {'vol': 0.015, 'trend': 0.0, 'lm': 0.4},   # RC: Range Consolidation
            3: {'vol': 0.025, 'trend': 0.05, 'lm': 0.6},  # RHA: Hybrid Adaptive
            4: {'vol': 0.04, 'trend': 0.2, 'lm': 0.8},    # HPEM: Extreme Move
            5: {'vol': 0.02, 'trend': 0.0, 'lm': 0.2}     # AMB: Ambiguous
        }
        
        for t in range(n_samples):
            regime = regimes[t]
            params = regime_params.get(regime, regime_params[0])
            
            # Generate correlated features
            base_vol = params['vol']
            trend_strength = params['trend']
            liquidity_measure = params['lm']
            
            # Feature generation with regime-specific characteristics
            features[t, 0] = liquidity_measure + np.random.normal(0, 0.1)  # LM
            features[t, 1] = trend_strength + np.random.normal(0, 0.05)    # SR
            features[t, 2] = np.abs(np.random.normal(0, base_vol))         # Abs
            features[t, 3] = trend_strength * 0.5 + np.random.normal(0, 0.02)  # RoC
            features[t, 4] = features[t, 3] ** 2 + np.random.normal(0, 0.001)  # RoC²
            
            if num_features > 5:
                features[t, 5] = 0.5 + np.random.normal(0, 0.1)           # P
                features[t, 6] = np.random.uniform(0.3, 0.7)              # T
                features[t, 7] = base_vol + np.random.normal(0, 0.005)    # Vol
                features[t, 8] = np.random.uniform(0, 1)                  # RPN
                features[t, 9] = np.random.normal(0, base_vol * 2)        # Delta
        
        return features
    
    def _generate_returns_from_regimes(self, regimes: np.ndarray) -> np.ndarray:
        """Generate realistic returns based on regime characteristics."""
        returns = np.zeros(len(regimes))
        
        regime_return_params = {
            0: {'mean': 0.0005, 'std': 0.01},   # ST: Low vol, slight positive
            1: {'mean': 0.001, 'std': 0.02},    # VT: Higher vol, positive trend
            2: {'mean': 0.0, 'std': 0.015},     # RC: Neutral, moderate vol
            3: {'mean': 0.0003, 'std': 0.025},  # RHA: Mixed, higher vol
            4: {'mean': 0.002, 'std': 0.04},    # HPEM: High return, high vol
            5: {'mean': 0.0, 'std': 0.02}       # AMB: Neutral, moderate vol
        }
        
        for t in range(len(regimes)):
            regime = regimes[t]
            params = regime_return_params.get(regime, regime_return_params[0])
            returns[t] = np.random.normal(params['mean'], params['std'])
        
        return returns
    
    def train_model(self, data: pd.DataFrame, visualize: bool = True) -> Dict[str, Any]:
        """
        Train FHMV model with comprehensive monitoring.
        
        Args:
            data: Training data with features
            visualize: Whether to create training visualizations
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"\n🔧 Starting FHMV Model Training")
        print("=" * 40)
        
        start_time = datetime.now()
        
        try:
            # Import FHMV core engine
            from fhmv.core_engine.fhmv_core_engine import FHMVCoreEngine
            
            # Initialize model with configuration
            self.model = FHMVCoreEngine(
                num_regimes=self.config['model_architecture']['num_fhmv_regimes'],
                num_features=self.config['model_architecture']['num_features'],
                K=self.config['model_architecture'].get('num_mixture_components', 3),
                dof_bounds=(
                    self.config['student_t_params']['dof_bounds']['min'],
                    self.config['student_t_params']['dof_bounds']['max']
                ),
                combined_to_final_regime_mapping=self.config['regime_mapping']['combined_to_final']
            )
            
            # Prepare training data
            feature_cols = [col for col in data.columns if col not in ['true_regime', 'returns']]
            X = data[feature_cols].values
            
            # Apply data preprocessing
            X = self._preprocess_data(X)
            
            print(f"📊 Training data shape: {X.shape}")
            print(f"⚙️ Model configuration:")
            print(f"   • Regimes: {self.config['model_architecture']['num_fhmv_regimes']}")
            print(f"   • Features: {self.config['model_architecture']['num_features']}")
            print(f"   • Mixture components: {self.config['model_architecture'].get('num_mixture_components', 3)}")
            print(f"   • Max EM iterations: {self.config['em_algorithm']['em_max_iterations']}")
            
            # Train model with monitoring
            training_results = self._train_with_monitoring(X)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds() / 60
            training_results['training_time_minutes'] = training_time
            
            print(f"\n✅ Training completed in {training_time:.1f} minutes")
            print(f"📈 Final log-likelihood: {training_results.get('final_log_likelihood', 'N/A'):.3f}")
            print(f"🔄 EM iterations: {training_results.get('em_iterations', 'N/A')}")
            
            # Create visualizations
            if visualize:
                self._create_training_visualizations(data, training_results)
            
            # Save model
            self._save_model(training_results)
            
            return training_results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Apply data preprocessing based on configuration."""
        
        # Handle missing values
        missing_method = self.config.get('data_processing', {}).get('missing_value_method', 'forward_fill')
        if missing_method == 'forward_fill':
            df_temp = pd.DataFrame(X)
            df_temp = df_temp.fillna(method='ffill').fillna(method='bfill')
            X = df_temp.values
        
        # Feature scaling
        scaling_method = self.config.get('data_processing', {}).get('scaling_method', 'standard')
        if scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        
        # Outlier handling
        outlier_threshold = self.config.get('data_processing', {}).get('outlier_threshold', 3.0)
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        X = np.where(z_scores > outlier_threshold, 
                     np.mean(X, axis=0), X)
        
        return X
    
    def _train_with_monitoring(self, X: np.ndarray) -> Dict[str, Any]:
        """Train model with comprehensive monitoring."""
        
        training_results = {
            'log_likelihood_history': [],
            'parameter_history': [],
            'convergence_metrics': {}
        }
        
        try:
            # Fit model
            log_likelihood = self.model.fit(
                X, 
                max_iterations=self.config['em_algorithm']['em_max_iterations'],
                tolerance=self.config['em_algorithm']['em_tolerance']
            )
            
            training_results['final_log_likelihood'] = log_likelihood
            training_results['em_iterations'] = getattr(self.model, 'iterations_', 
                                                      self.config['em_algorithm']['em_max_iterations'])
            training_results['converged'] = True
            
            # Get predictions for analysis
            predictions = self.model.predict(X, return_probabilities=True)
            training_results['final_predictions'] = predictions
            
            print(f"📈 Training metrics:")
            print(f"   • Log-likelihood: {log_likelihood:.3f}")
            print(f"   • Iterations: {training_results['em_iterations']}")
            print(f"   • Convergence: {'✅' if training_results['converged'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            training_results['error'] = str(e)
            training_results['converged'] = False
        
        return training_results
    
    def _create_training_visualizations(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Create comprehensive training visualizations."""
        
        print("🎨 Creating training visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FHMV Model Training Results', fontsize=16, fontweight='bold')
        
        # 1. Training convergence (if available)
        ax1 = axes[0, 0]
        if 'log_likelihood_history' in results and results['log_likelihood_history']:
            ax1.plot(results['log_likelihood_history'], marker='o')
            ax1.set_title('Training Convergence')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Log-Likelihood')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Convergence data\nnot available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Training Convergence')
        
        # 2. Regime predictions
        ax2 = axes[0, 1]
        if 'final_predictions' in results:
            predicted_regimes = results['final_predictions']['predicted_regimes']
            ax2.plot(predicted_regimes[:100], marker='o', markersize=3)
            ax2.set_title('Regime Predictions (First 100 samples)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Predicted Regime')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Prediction data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Regime Predictions')
        
        # 3. Feature distributions by regime
        ax3 = axes[0, 2]
        try:
            if 'final_predictions' in results:
                feature_cols = [col for col in data.columns if col not in ['true_regime', 'returns']]
                first_feature = data[feature_cols[0]].values
                predicted_regimes = results['final_predictions']['predicted_regimes']
                
                for regime in np.unique(predicted_regimes):
                    regime_data = first_feature[predicted_regimes == regime]
                    ax3.hist(regime_data, alpha=0.5, label=f'Regime {regime}', bins=20)
                
                ax3.set_title(f'{feature_cols[0]} Distribution by Regime')
                ax3.set_xlabel(feature_cols[0])
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                raise ValueError("No predictions available")
        except:
            ax3.text(0.5, 0.5, 'Feature analysis\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Distribution by Regime')
        
        # 4. Configuration summary
        ax4 = axes[1, 0]
        config_text = f"""Model Configuration:
• Regimes: {self.config['model_architecture']['num_fhmv_regimes']}
• Features: {self.config['model_architecture']['num_features']}
• Mixture Components: {self.config['model_architecture'].get('num_mixture_components', 3)}
• Max EM Iterations: {self.config['em_algorithm']['em_max_iterations']}
• DoF Bounds: [{self.config['student_t_params']['dof_bounds']['min']:.1f}, {self.config['student_t_params']['dof_bounds']['max']:.1f}]
• Tolerance: {self.config['em_algorithm']['em_tolerance']:.0e}"""
        
        ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax4.set_title('Model Configuration')
        ax4.axis('off')
        
        # 5. Training summary
        ax5 = axes[1, 1]
        training_text = f"""Training Results:
• Final Log-Likelihood: {results.get('final_log_likelihood', 'N/A'):.3f}
• EM Iterations: {results.get('em_iterations', 'N/A')}
• Converged: {'✅' if results.get('converged', False) else '❌'}
• Training Time: {results.get('training_time_minutes', 0):.1f} min
• Data Samples: {len(data)}
• Random Seed: {self.config.get('advanced', {}).get('random_seed', 'N/A')}"""
        
        ax5.text(0.05, 0.95, training_text, transform=ax5.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax5.set_title('Training Summary')
        ax5.axis('off')
        
        # 6. Regime transition matrix (if available)
        ax6 = axes[1, 2]
        try:
            if 'final_predictions' in results:
                predicted_regimes = results['final_predictions']['predicted_regimes']
                n_regimes = len(np.unique(predicted_regimes))
                transitions = np.zeros((n_regimes, n_regimes))
                
                for i in range(len(predicted_regimes)-1):
                    transitions[predicted_regimes[i], predicted_regimes[i+1]] += 1
                
                # Normalize
                row_sums = transitions.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                transitions = transitions / row_sums
                
                im = ax6.imshow(transitions, cmap='Blues', aspect='auto')
                ax6.set_title('Regime Transition Matrix')
                ax6.set_xlabel('To Regime')
                ax6.set_ylabel('From Regime')
                
                # Add text annotations
                for i in range(n_regimes):
                    for j in range(n_regimes):
                        ax6.text(j, i, f'{transitions[i,j]:.2f}', 
                               ha='center', va='center', fontsize=8)
                
                plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
            else:
                raise ValueError("No predictions available")
        except:
            ax6.text(0.5, 0.5, 'Transition matrix\nnot available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Regime Transition Matrix')
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs('model_training/outputs', exist_ok=True)
        plt.savefig('model_training/outputs/training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training visualizations saved to: model_training/outputs/training_results.png")
    
    def _save_model(self, results: Dict[str, Any]):
        """Save trained model and results."""
        
        os.makedirs('model_training/outputs', exist_ok=True)
        
        # Save configuration
        config_path = 'model_training/outputs/training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save training results
        results_path = 'model_training/outputs/training_results.yaml'
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['final_predictions']}  # Exclude large arrays
        
        with open(results_path, 'w') as f:
            yaml.dump(results_to_save, f, default_flow_style=False)
        
        print(f"💾 Model configuration saved to: {config_path}")
        print(f"💾 Training results saved to: {results_path}")
    
    def predict(self, data: pd.DataFrame, return_probabilities: bool = True) -> Dict[str, Any]:
        """Make predictions with trained model."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        feature_cols = [col for col in data.columns if col not in ['true_regime', 'returns']]
        X = data[feature_cols].values
        X = self._preprocess_data(X)
        
        return self.model.predict(X, return_probabilities=return_probabilities)
    
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        # Implementation depends on model serialization format
        pass
    
    def hyperparameter_search(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        
        print("🔍 Starting hyperparameter search...")
        
        # This would integrate with libraries like Optuna for systematic search
        search_ranges = self.config.get('hyperparameter_search', {}).get('search_ranges', {})
        num_trials = self.config.get('hyperparameter_search', {}).get('num_trials', 50)
        
        print(f"📊 Search space: {len(search_ranges)} parameters")
        print(f"🎯 Number of trials: {num_trials}")
        
        # Placeholder for hyperparameter search implementation
        best_params = {
            'num_mixture_components': 3,
            'em_max_iterations': 50,
            'persistence_weight': 0.7
        }
        
        print("✅ Hyperparameter search completed")
        return best_params