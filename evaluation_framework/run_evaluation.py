#!/usr/bin/env python3
"""
FHMV Unsupervised Evaluation - Main Runner
==========================================

Clean, focused evaluation script for FHMV models without ground truth.
Includes BTC price visualization with regime predictions.

Usage: python evaluation_framework/run_evaluation.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run streamlined unsupervised evaluation."""
    print("🎯 FHMV Unsupervised Evaluation")
    print("=" * 40)
    
    try:
        # Import required modules
        from train_fhmv_model import FHMVTrainingPipeline
        from evaluation_framework.unsupervised_evaluator import StreamlinedUnsupervisedEvaluator
        
        # Configuration for evaluation
        config = {
            "num_fhmv_regimes": 6,
            "num_features": 10,
            "em_max_iterations": 8,
            "n_samples": 365,  # One year of daily data
            "combined_to_final_regime_mapping": {
                0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0, 7: 1
            }
        }
        
        print("⚙️  Training FHMV model...")
        pipeline = FHMVTrainingPipeline(config)
        
        # Generate market data (simulating BTC data)
        data = pipeline.generate_synthetic_data(n_samples=config['n_samples'])
        
        # Train model
        pipeline.train_model(data, visualize=False)
        print("✅ Model training completed")
        
        # Initialize evaluator
        regime_names = ['ST', 'VT', 'RC', 'RHA', 'HPEM', 'AMB']
        evaluator = StreamlinedUnsupervisedEvaluator(pipeline, regime_names)
        
        # Run evaluation
        print("\n🔍 Running unsupervised evaluation...")
        results = evaluator.evaluate_model_quality(data, save_plots=True)
        
        # Display results
        print("\n📊 EVALUATION RESULTS")
        print("-" * 25)
        
        # Overall quality score
        quality_score = results['composite_score']
        print(f"🎯 Overall Quality Score: {quality_score:.3f}/1.000")
        
        # Quality assessment
        if quality_score >= 0.8:
            assessment = "🟢 EXCELLENT - Production ready"
        elif quality_score >= 0.6:
            assessment = "🟡 GOOD - Minor improvements needed"
        elif quality_score >= 0.4:
            assessment = "🟠 ACCEPTABLE - Needs improvements"
        else:
            assessment = "🔴 POOR - Major revisions required"
        
        print(f"📈 Assessment: {assessment}")
        
        # Key metrics breakdown
        print(f"\n📋 KEY METRICS")
        print(f"   • Prediction Confidence: {results['model_confidence']['mean_confidence']:.3f}")
        print(f"   • High Confidence Ratio: {results['model_confidence']['high_confidence_ratio']:.1%}")
        print(f"   • Regime Separation (Silhouette): {results['regime_quality']['silhouette_score']:.3f}")
        print(f"   • Economic Sharpe Ratio: {results['economic_value']['sharpe_ratio']:.3f}")
        print(f"   • Regime Stability: {results['stability_metrics']['regime_stability']:.3f}")
        print(f"   • Avg Regime Duration: {results['stability_metrics']['mean_duration']:.1f} periods")
        
        # Performance insights
        print(f"\n💡 INSIGHTS")
        print("-" * 12)
        
        confidence = results['model_confidence']['mean_confidence']
        if confidence > 0.8:
            print("✅ High confidence predictions indicate reliable regime identification")
        elif confidence > 0.6:
            print("⚠️ Moderate confidence - consider more training data")
        else:
            print("❌ Low confidence - model needs improvement")
        
        silhouette = results['regime_quality']['silhouette_score']
        if silhouette > 0.5:
            print("✅ Excellent regime separation quality")
        elif silhouette > 0.3:
            print("⚠️ Good regime separation with room for improvement")
        else:
            print("❌ Poor regime separation - check feature engineering")
        
        sharpe = results['economic_value']['sharpe_ratio']
        if sharpe > 1.0:
            print("✅ Strong economic value for trading strategies")
        elif sharpe > 0.5:
            print("⚠️ Moderate economic value")
        else:
            print("❌ Weak economic performance")
        
        # Files generated
        print(f"\n📁 OUTPUT FILES")
        print("-" * 15)
        output_file = "evaluation_framework/outputs/unsupervised_evaluation_dashboard.png"
        if os.path.exists(output_file):
            print(f"✅ {output_file}")
            print("   • BTC price chart with regime predictions")
            print("   • Confidence distribution analysis")
            print("   • Economic performance metrics")
            print("   • Model health indicators")
        
        # Recommendations
        print(f"\n🚀 RECOMMENDATIONS")
        print("-" * 18)
        
        if quality_score >= 0.7:
            print("• Model is ready for production deployment")
            print("• Set up continuous monitoring")
            print("• Consider A/B testing with current strategy")
        elif quality_score >= 0.5:
            print("• Focus on improving prediction confidence")
            print("• Enhance feature engineering")
            print("• Increase training data quality")
        else:
            print("• Review model architecture")
            print("• Check data preprocessing pipeline")
            print("• Consider different parameter settings")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 View dashboard: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()