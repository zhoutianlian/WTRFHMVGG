import os
import yaml
import json
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Configuration manager for BTC feature engineering pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    elif self.config_path.endswith('.json'):
                        config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {self.config_path}")
                
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()
        
        except Exception as e:
            self.logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'database': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'btc_analysis'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'postgres')
            },
            'glassnode': {
                'api_key': os.getenv('GLASSNODE_API_KEY', ''),
                'base_url': 'https://api.glassnode.com/v1/',
                'requests_per_minute': 10
            },
            'logging': {
                'level': 'INFO',
                'file': 'btc_pipeline.log',
                'max_size_mb': 100,
                'backup_count': 5
            },
            'processing': {
                'batch_size': 10000,
                'default_days_back': 30,
                'max_age_hours': 2
            },
            'features': {
                'rpn': {
                    'wavelet': 'coif3',
                    'level': 6,
                    'ema_span_diff': 16,
                    'ema_span_lld': 12,
                    'extreme_window_hours': 720
                },
                'bin': {
                    'n_clusters': 9,
                    'momentum_window_hours': 720
                },
                'dominance': {
                    'classification_enabled': True
                },
                'signals': {
                    'beta_window': 8,
                    'ceiling_bottom_enabled': True,
                    'reversal_enabled': True
                }
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation)
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key (supports dot notation)
        
        Args:
            key: Configuration key (e.g., 'database.host')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            output_path: Path to save config. If None, uses original path.
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w') as f:
                if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif output_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {output_path}")
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            raise
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration and return validation results
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate database config
        db_config = self.get('database', {})
        required_db_fields = ['host', 'port', 'database', 'user', 'password']
        
        for field in required_db_fields:
            if not db_config.get(field):
                results['errors'].append(f"Missing database.{field}")
                results['valid'] = False
        
        # Validate Glassnode config
        glassnode_config = self.get('glassnode', {})
        if not glassnode_config.get('api_key'):
            results['warnings'].append("Missing glassnode.api_key - some features may not work")
        
        # Validate processing config
        processing_config = self.get('processing', {})
        batch_size = processing_config.get('batch_size', 10000)
        if not isinstance(batch_size, int) or batch_size <= 0:
            results['errors'].append("processing.batch_size must be a positive integer")
            results['valid'] = False
        
        # Validate feature config
        features_config = self.get('features', {})
        
        # RPN config
        rpn_config = features_config.get('rpn', {})
        if rpn_config.get('level', 6) > 10:
            results['warnings'].append("features.rpn.level > 10 may cause performance issues")
        
        # Bin config
        bin_config = features_config.get('bin', {})
        n_clusters = bin_config.get('n_clusters', 9)
        if not isinstance(n_clusters, int) or n_clusters < 2:
            results['errors'].append("features.bin.n_clusters must be >= 2")
            results['valid'] = False
        
        return results
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get('database', {})
    
    def get_glassnode_config(self) -> Dict[str, Any]:
        """Get Glassnode API configuration"""
        return self.get('glassnode', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.get('processing', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get features configuration"""
        return self.get('features', {})
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        # Database config from environment
        if os.getenv('DB_HOST'):
            self.set('database.host', os.getenv('DB_HOST'))
        if os.getenv('DB_PORT'):
            self.set('database.port', int(os.getenv('DB_PORT')))
        if os.getenv('DB_NAME'):
            self.set('database.database', os.getenv('DB_NAME'))
        if os.getenv('DB_USER'):
            self.set('database.user', os.getenv('DB_USER'))
        if os.getenv('DB_PASSWORD'):
            self.set('database.password', os.getenv('DB_PASSWORD'))
        
        # Glassnode config from environment
        if os.getenv('GLASSNODE_API_KEY'):
            self.set('glassnode.api_key', os.getenv('GLASSNODE_API_KEY'))
        
        # Logging config from environment
        if os.getenv('LOG_LEVEL'):
            self.set('logging.level', os.getenv('LOG_LEVEL'))
        if os.getenv('LOG_FILE'):
            self.set('logging.file', os.getenv('LOG_FILE'))
        
        self.logger.info("Configuration updated from environment variables")
    
    def create_sample_config(self, output_path: str):
        """
        Create a sample configuration file
        
        Args:
            output_path: Path where to create the sample config
        """
        sample_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'btc_analysis',
                'user': 'postgres',
                'password': 'your_password_here'
            },
            'glassnode': {
                'api_key': 'your_glassnode_api_key_here',
                'base_url': 'https://api.glassnode.com/v1/',
                'requests_per_minute': 10
            },
            'logging': {
                'level': 'INFO',
                'file': 'btc_pipeline.log',
                'max_size_mb': 100,
                'backup_count': 5
            },
            'processing': {
                'batch_size': 10000,
                'default_days_back': 30,
                'max_age_hours': 2
            },
            'features': {
                'rpn': {
                    'wavelet': 'coif3',
                    'level': 6,
                    'ema_span_diff': 16,
                    'ema_span_lld': 12,
                    'extreme_window_hours': 720
                },
                'bin': {
                    'n_clusters': 9,
                    'momentum_window_hours': 720
                },
                'dominance': {
                    'classification_enabled': True
                },
                'signals': {
                    'beta_window': 8,
                    'ceiling_bottom_enabled': True,
                    'reversal_enabled': True
                }
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            
            print(f"Sample configuration created at {output_path}")
            print("Please edit the configuration file with your settings before running the pipeline.")
            
        except Exception as e:
            print(f"Error creating sample config: {e}")
            raise