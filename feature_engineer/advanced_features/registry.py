"""
Advanced Features Registry

This module manages the registry of all available advanced features
and provides utilities for feature discovery and instantiation.
"""

from typing import Dict, List, Any, Type, Optional
import logging
from .base import BaseAdvancedFeature
from .volatility import AdaptiveVolatilityFeature, KAMAFeature, GARCHVolatilityFeature
from .momentum import ROCDynamicsFeature, MomentumIndicatorsFeature, MACDFeature
from .statistical import SpikeRatioFeature, KalmanSlopeFeature, StatisticalMomentsFeature
# Import existing features
from .rpn_features import (
    CWTDetrendingFeature, KalmanFilteringFeature, EMAFeature, 
    RPNExtremeFeature, CorrelationAnalysisFeature, RPNCalculationFeature
)
from .binning_features import (
    KMeansBinningFeature, BinMomentumFeature, BinAnalysisFeature
)
from .dominance_features import (
    DominanceCalculationFeature, DominanceStatusFeature, 
    DominanceClassificationFeature, MarketRegimeFeature
)
from .signal_features import (
    BetaCalculationFeature, CeilingBottomFeature, ReversalSignalFeature,
    SignalGenerationFeature, SignalAnalysisFeature
)


class AdvancedFeaturesRegistry:
    """Registry for managing advanced features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._features: Dict[str, BaseAdvancedFeature] = {}
        self._feature_classes: Dict[str, Type[BaseAdvancedFeature]] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the registry with predefined advanced features"""
        
        # =================================
        # EXISTING FEATURES (Priority 0-3)
        # These must run first as they recreate the original feature pipeline
        # =================================
        
        # Priority 0: RPN Features (base features from raw data)
        self.register_feature('cwt_detrending', CWTDetrendingFeature(
            input_columns=['lld_normal', 'fll_normal', 'fsl_normal'],
            wavelet='coif3',
            level=6
        ))
        
        self.register_feature('kalman_filtering', KalmanFilteringFeature(
            input_columns=['lld_cwt', 'fll_cwt', 'fsl_cwt'],
            covariance_params={'initial': 50, 'observation': 80, 'transition': 0.05}
        ))
        
        self.register_feature('ema_smoothing', EMAFeature(
            input_columns=['diff_ls_cwt_kf', 'lld_cwt_kf'],
            ema_configs=[
                {'input_col': 'diff_ls_cwt_kf', 'output_col': 'diff_ls_smooth', 'span': 16},
                {'input_col': 'lld_cwt_kf', 'output_col': 'lld_cwt_kf_smooth', 'span': 12}
            ]
        ))
        
        self.register_feature('rpn_extreme', RPNExtremeFeature(
            input_col='lld_cwt_kf',
            result_col='is_rpn_extreme',
            window_size=30 * 24
        ))
        
        self.register_feature('correlation_analysis', CorrelationAnalysisFeature(
            fll_col='fll_cwt_kf',
            fsl_col='fsl_cwt_kf'
        ))
        
        self.register_feature('rpn_calculation', RPNCalculationFeature(
            primary_col='lld_cwt_kf',
            fallback_col='lld_normal'
        ))
        
        # Priority 1: Binning Features (depends on RPN)
        self.register_feature('kmeans_binning', KMeansBinningFeature(
            rpn_col='risk_priority_number',
            n_clusters=9
        ))
        
        self.register_feature('bin_momentum', BinMomentumFeature(
            bin_col='bin_index',
            window_size=24 * 30
        ))
        
        self.register_feature('bin_analysis', BinAnalysisFeature(
            bin_col='bin_index'
        ))
        
        # Priority 2: Dominance Features (depends on RPN + Binning)
        self.register_feature('dominance_calculation', DominanceCalculationFeature(
            rpn_col='risk_priority_number',
            bin_col='bin_index',
            diff_col='diff_ls_cwt_kf'
        ))
        
        self.register_feature('dominance_status', DominanceStatusFeature(
            dominance_col='dominance'
        ))
        
        self.register_feature('dominance_classification', DominanceClassificationFeature(
            dominance_col='dominance'
        ))
        
        self.register_feature('market_regime', MarketRegimeFeature(
            dominance_col='dominance',
            bin_col='bin_index'
        ))
        
        # Priority 3: Signal Features (depends on all previous)
        self.register_feature('beta_calculation', BetaCalculationFeature(
            input_col='diff_ls_cwt_kf',
            result_col='diff_beta_8',
            window_size=8
        ))
        
        self.register_feature('ceiling_bottom', CeilingBottomFeature(
            dominance_col='dominance',
            bin_col='bin_index',
            diff_col='diff_ls_cwt_kf',
            fll_col='fll_cwt_kf',
            fsl_col='fsl_cwt_kf'
        ))
        
        self.register_feature('reversal_signal', ReversalSignalFeature(
            cb_col='hit_ceiling_bottom',
            beta_col='diff_beta_8',
            diff_col='diff_ls_cwt_kf'
        ))
        
        self.register_feature('signal_generation', SignalGenerationFeature(
            dominance_col='dominance',
            cb_col='hit_ceiling_bottom',
            reversal_col='reverse_ceiling_bottom'
        ))
        
        self.register_feature('signal_analysis', SignalAnalysisFeature(
            signal_class_col='signal_class',
            price_col='price'
        ))
        
        # =================================
        # ADVANCED FEATURES (Priority 4+)
        # These run after all existing features are calculated
        # =================================
        
        # Volatility features that work with existing price/liquidation data
        self.register_feature('adaptive_volatility', AdaptiveVolatilityFeature(
            price_col='price',
            short_window=6,
            long_window=24,
            result_col='vol_adaptive'
        ))
        
        # KAMA features for existing filtered data
        self.register_feature('kama_fll', KAMAFeature(
            value_col='fll_cwt_kf',
            window=10,
            result_col='fll_kama'
        ))
        
        self.register_feature('kama_fsl', KAMAFeature(
            value_col='fsl_cwt_kf', 
            window=10,
            result_col='fsl_kama'
        ))
        
        self.register_feature('kama_price', KAMAFeature(
            value_col='price',
            window=14,
            result_col='price_kama'
        ))
        
        # Spike ratio features
        self.register_feature('spike_ratio_fll', SpikeRatioFeature(
            value_col='fll_cwt_kf',
            window=24,
            method='kama',
            spike_ratio_col='fll_spike_kama'
        ))
        
        self.register_feature('spike_ratio_fsl', SpikeRatioFeature(
            value_col='fsl_cwt_kf',
            window=24,
            method='kama',
            spike_ratio_col='fsl_spike_kama'
        ))
        
        self.register_feature('spike_ratio_price', SpikeRatioFeature(
            value_col='price',
            window=20,
            method='ema',
            spike_ratio_col='price_spike_ema'
        ))
        
        # ROC dynamics features
        self.register_feature('roc_dynamics_fll', ROCDynamicsFeature(
            value_col='fll_cwt_kf',
            velocity_col='fll_velocity',
            acceleration_col='fll_acceleration',
            smoothing_method='gaussian',
            window=13
        ))
        
        self.register_feature('roc_dynamics_price', ROCDynamicsFeature(
            value_col='price',
            velocity_col='price_velocity',
            acceleration_col='price_acceleration',
            smoothing_method='savgol',
            window=7
        ))
        
        # Momentum indicators
        self.register_feature('momentum_indicators', MomentumIndicatorsFeature(
            value_col='price',
            window=14,
            include_rsi=True,
            include_stoch=True
        ))
        
        # MACD
        self.register_feature('macd_price', MACDFeature(
            value_col='price',
            fast_period=12,
            slow_period=26,
            signal_period=9,
            result_col='price_macd'
        ))
        
        # Kalman slope estimation
        self.register_feature('kalman_slope', KalmanSlopeFeature(
            price_col='price',
            process_noise=0.01,
            measurement_noise=1.0,
            result_col='kalman_slope'
        ))
        
        # Statistical moments
        self.register_feature('statistical_moments', StatisticalMomentsFeature(
            value_col='price',
            window=24,
            include_skewness=True,
            include_kurtosis=True,
            include_jarque_bera=True
        ))
        
        # GARCH volatility
        self.register_feature('garch_volatility', GARCHVolatilityFeature(
            price_col='price',
            alpha=0.1,
            beta=0.85,
            omega=0.05,
            result_col='vol_garch'
        ))
        
        # Register feature classes for dynamic instantiation
        self._feature_classes.update({
            # Existing feature classes
            'cwt_detrending': CWTDetrendingFeature,
            'kalman_filtering': KalmanFilteringFeature,
            'ema_smoothing': EMAFeature,
            'rpn_extreme': RPNExtremeFeature,
            'correlation_analysis': CorrelationAnalysisFeature,
            'rpn_calculation': RPNCalculationFeature,
            'kmeans_binning': KMeansBinningFeature,
            'bin_momentum': BinMomentumFeature,
            'bin_analysis': BinAnalysisFeature,
            'dominance_calculation': DominanceCalculationFeature,
            'dominance_status': DominanceStatusFeature,
            'dominance_classification': DominanceClassificationFeature,
            'market_regime': MarketRegimeFeature,
            'beta_calculation': BetaCalculationFeature,
            'ceiling_bottom': CeilingBottomFeature,
            'reversal_signal': ReversalSignalFeature,
            'signal_generation': SignalGenerationFeature,
            'signal_analysis': SignalAnalysisFeature,
            # Advanced feature classes
            'adaptive_volatility': AdaptiveVolatilityFeature,
            'kama': KAMAFeature,
            'garch_volatility': GARCHVolatilityFeature,
            'roc_dynamics': ROCDynamicsFeature,
            'momentum_indicators': MomentumIndicatorsFeature,
            'macd': MACDFeature,
            'spike_ratio': SpikeRatioFeature,
            'kalman_slope': KalmanSlopeFeature,
            'statistical_moments': StatisticalMomentsFeature
        })
        
        self.logger.info(f"Initialized advanced features registry with {len(self._features)} features")
    
    def register_feature(self, name: str, feature: BaseAdvancedFeature):
        """Register a new feature instance"""
        self._features[name] = feature
        self.logger.debug(f"Registered feature: {name}")
    
    def get_feature(self, name: str) -> Optional[BaseAdvancedFeature]:
        """Get a feature instance by name"""
        return self._features.get(name)
    
    def list_features(self) -> List[str]:
        """List all available feature names"""
        return list(self._features.keys())
    
    def get_features_by_category(self) -> Dict[str, List[str]]:
        """Group features by category"""
        categories = {
            'volatility': [],
            'momentum': [],
            'statistical': [],
            'trend': []
        }
        
        for name, feature in self._features.items():
            if 'vol' in name or 'garch' in name:
                categories['volatility'].append(name)
            elif 'rsi' in name or 'stoch' in name or 'macd' in name or 'roc' in name or 'momentum' in name:
                categories['momentum'].append(name)
            elif 'spike' in name or 'kalman' in name or 'moments' in name:
                categories['statistical'].append(name)
            elif 'kama' in name or 'slope' in name:
                categories['trend'].append(name)
            else:
                # Default to statistical
                categories['statistical'].append(name)
        
        return categories
    
    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """Get the required columns for a feature"""
        feature = self.get_feature(feature_name)
        if feature:
            return feature.required_columns
        return []
    
    def validate_feature_dependencies(self, feature_names: List[str], available_columns: List[str]) -> Dict[str, List[str]]:
        """
        Validate that all feature dependencies are met
        
        Returns:
            Dict mapping feature names to missing dependencies
        """
        missing_deps = {}
        
        for feature_name in feature_names:
            feature = self.get_feature(feature_name)
            if feature:
                missing = [col for col in feature.required_columns if col not in available_columns]
                if missing:
                    missing_deps[feature_name] = missing
        
        return missing_deps
    
    def get_execution_order(self, feature_names: List[str]) -> List[str]:
        """
        Determine optimal execution order based on dependencies
        
        Features that depend on existing columns should run first,
        followed by features that might depend on computed features.
        """
        # Execution order: existing features MUST run first, then advanced features
        priority_order = {
            # Priority 0: RPN Features (base processing from raw data)
            'cwt_detrending': 0,
            'kalman_filtering': 0,
            'ema_smoothing': 0,
            'rpn_extreme': 0,
            'correlation_analysis': 0,
            'rpn_calculation': 0,
            
            # Priority 1: Binning Features (depends on RPN)
            'kmeans_binning': 1,
            'bin_momentum': 1,
            'bin_analysis': 1,
            
            # Priority 2: Dominance Features (depends on RPN + Binning)
            'dominance_calculation': 2,
            'dominance_status': 2,
            'dominance_classification': 2,
            'market_regime': 2,
            
            # Priority 3: Signal Features (depends on all previous)
            'beta_calculation': 3,
            'ceiling_bottom': 3,
            'reversal_signal': 3,
            'signal_generation': 3,
            'signal_analysis': 3,
            
            # Priority 4+: Advanced Features (run after all existing features)
            'adaptive_volatility': 4,
            'garch_volatility': 4,
            'kalman_slope': 4,
            
            # KAMA features (depend on existing filtered data)
            'kama_fll': 5,
            'kama_fsl': 5, 
            'kama_price': 5,
            
            # Momentum indicators
            'momentum_indicators': 6,
            'macd_price': 6,
            
            # Spike ratios (may depend on KAMA)
            'spike_ratio_fll': 7,
            'spike_ratio_fsl': 7,
            'spike_ratio_price': 7,
            
            # ROC dynamics
            'roc_dynamics_fll': 8,
            'roc_dynamics_price': 8,
            
            # Statistical features last
            'statistical_moments': 9
        }
        
        # Sort by priority, then alphabetically
        sorted_features = sorted(
            feature_names,
            key=lambda x: (priority_order.get(x, 999), x)
        )
        
        return sorted_features
    
    def create_feature(self, feature_class: str, **kwargs) -> Optional[BaseAdvancedFeature]:
        """Create a new feature instance dynamically"""
        if feature_class in self._feature_classes:
            try:
                return self._feature_classes[feature_class](**kwargs)
            except Exception as e:
                self.logger.error(f"Failed to create feature {feature_class}: {e}")
                return None
        return None
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a feature"""
        feature = self.get_feature(feature_name)
        if not feature:
            return {'error': f'Feature {feature_name} not found'}
        
        return {
            'name': feature_name,
            'feature_type': feature.feature_name,
            'parameters': feature.parameters,
            'required_columns': feature.required_columns,
            'class': feature.__class__.__name__,
            'description': feature.__class__.__doc__ or 'No description available'
        }
    
    def get_all_features_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered features"""
        return {name: self.get_feature_info(name) for name in self.list_features()}


# Create global registry instance
ADVANCED_FEATURES_REGISTRY = AdvancedFeaturesRegistry()


def get_registry() -> AdvancedFeaturesRegistry:
    """Get the global advanced features registry"""
    return ADVANCED_FEATURES_REGISTRY


def list_available_features() -> List[str]:
    """Convenience function to list all available features"""
    return ADVANCED_FEATURES_REGISTRY.list_features()


def get_feature_by_name(name: str) -> Optional[BaseAdvancedFeature]:
    """Convenience function to get a feature by name"""
    return ADVANCED_FEATURES_REGISTRY.get_feature(name)


def get_features_by_category() -> Dict[str, List[str]]:
    """Convenience function to get features grouped by category"""
    return ADVANCED_FEATURES_REGISTRY.get_features_by_category()


# Example usage and feature descriptions
FEATURE_DESCRIPTIONS = {
    # Existing Features (recreate original pipeline)
    'cwt_detrending': 'Continuous Wavelet Transform detrending for liquidation data normalization',
    'kalman_filtering': 'Kalman filter smoothing of CWT-detrended liquidation signals',
    'ema_smoothing': 'Exponential Moving Average smoothing for liquidation differentials and RPN',
    'rpn_extreme': 'Risk Priority Number extreme condition detection based on bin patterns',
    'correlation_analysis': 'Liquidation correlation pattern classification (FLL vs FSL dynamics)',
    'rpn_calculation': 'Risk Priority Number calculation from processed liquidation data',
    'kmeans_binning': 'K-means clustering (9 bins) of Risk Priority Number for market classification',
    'bin_momentum': 'Momentum analysis of bin index changes and trend persistence',
    'bin_analysis': 'Comprehensive binning analysis including regime classification',
    'dominance_calculation': 'Market dominance determination based on bin extremes and liquidation patterns',
    'dominance_status': 'Dominance persistence tracking and temporal analysis',
    'dominance_classification': 'Classification of dominance into Bull/Bear/Congestion categories',
    'market_regime': 'Market regime analysis combining dominance with bin-based classifications',
    'beta_calculation': 'Beta coefficient calculation for trend analysis and signal generation',
    'ceiling_bottom': 'Ceiling and bottom hit detection based on dominance and bin extremes',
    'reversal_signal': 'Reversal signal detection from ceiling/bottom hits and momentum analysis',
    'signal_generation': 'Multi-level trading signal generation (L1: dominance, L2: ceiling/bottom, L3: reversals)',
    'signal_analysis': 'Signal performance analysis, frequency tracking, and quality assessment',
    
    # Advanced Features (extend original functionality)
    'adaptive_volatility': 'Volatility that adapts to market regime changes by combining short and long-term volatilities',
    'kama_fll': 'Kaufman Adaptive Moving Average applied to Futures Long Liquidations',
    'kama_fsl': 'Kaufman Adaptive Moving Average applied to Futures Short Liquidations',
    'kama_price': 'Kaufman Adaptive Moving Average applied to price data',
    'spike_ratio_fll': 'Ratio of FLL to its KAMA, identifying liquidation spikes',
    'spike_ratio_fsl': 'Ratio of FSL to its KAMA, identifying liquidation spikes',
    'spike_ratio_price': 'Ratio of price to its EMA, identifying price spikes',
    'roc_dynamics_fll': 'Rate of change dynamics (velocity/acceleration) for FLL',
    'roc_dynamics_price': 'Rate of change dynamics (velocity/acceleration) for price',
    'momentum_indicators': 'Traditional RSI and Stochastic Oscillator indicators',
    'macd_price': 'Moving Average Convergence Divergence for price data',
    'kalman_slope': 'Kalman filter-based trend slope estimation',
    'statistical_moments': 'Rolling statistical moments (skewness, kurtosis)',
    'garch_volatility': 'GARCH-like volatility estimation with clustering effects'
}