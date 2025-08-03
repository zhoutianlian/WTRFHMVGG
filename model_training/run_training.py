#!/usr/bin/env python3
"""
FHMV Model Training Runner
=========================

Simple script to run FHMV model training with configuration management.

Usage:
    python model_training/run_training.py
    python model_training/run_training.py --config model_training/fhmv_config.yaml
"""

import os
import sys
import argparse
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main training runner."""
    parser = argparse.ArgumentParser(description='FHMV Model Training')
    parser.add_argument('--config', type=str, default='model_training/fhmv_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data (CSV file)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create training visualizations')
    parser.add_argument('--hyperopt', action='store_true', default=False,
                       help='Run hyperparameter optimization')
    
    args = parser.parse_args()
    
    print("🚀 FHMV Model Training")
    print("=" * 30)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ Config file: {args.config}")
    
    try:
        # Import training pipeline
        from model_training.training_pipeline import FHMVTrainingPipeline
        
        # Initialize pipeline
        pipeline = FHMVTrainingPipeline(config_path=args.config)
        
        # Load or generate data
        if args.data and os.path.exists(args.data):
            import pandas as pd
            print(f"📊 Loading data from: {args.data}")
            data = pd.read_csv(args.data)
        else:
            print("📊 Generating synthetic training data...")
            data = pipeline.generate_synthetic_data(n_samples=args.samples)
        
        print(f"📈 Training data shape: {data.shape}")
        
        # Run hyperparameter optimization if requested
        if args.hyperopt:
            print("\n🔍 Running hyperparameter optimization...")
            best_params = pipeline.hyperparameter_search(data)
            print(f"✅ Best parameters: {best_params}")
        
        # Train model
        print("\n🔧 Training FHMV model...")
        results = pipeline.train_model(data, visualize=args.visualize)
        
        # Display results
        if 'error' not in results:
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Results summary:")
            print(f"   • Final log-likelihood: {results.get('final_log_likelihood', 'N/A'):.3f}")
            print(f"   • Training time: {results.get('training_time_minutes', 0):.1f} minutes")
            print(f"   • EM iterations: {results.get('em_iterations', 'N/A')}")
            print(f"   • Converged: {'✅' if results.get('converged', False) else '❌'}")
            
            # Check if outputs were created
            outputs_dir = 'model_training/outputs'
            if os.path.exists(outputs_dir):
                files = os.listdir(outputs_dir)
                print(f"\n📁 Generated files:")
                for file in files:
                    print(f"   • {os.path.join(outputs_dir, file)}")
            
            # Quality assessment
            log_likelihood = results.get('final_log_likelihood', -np.inf)
            if log_likelihood > -2.0:
                quality = "🟢 EXCELLENT"
            elif log_likelihood > -3.0:
                quality = "🟡 GOOD"
            elif log_likelihood > -5.0:
                quality = "🟠 ACCEPTABLE"
            else:
                quality = "🔴 POOR"
            
            print(f"\n🎯 Model Quality: {quality}")
            
        else:
            print(f"❌ Training failed: {results['error']}")
            return 1
        
        print(f"\n📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
        
    except Exception as e:
        print(f"❌ Training script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import numpy as np
    exit_code = main()
    sys.exit(exit_code)