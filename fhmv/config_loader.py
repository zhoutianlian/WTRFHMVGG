# fhmv/config_loader.py

import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """
    Utility class for loading system configurations, typically from YAML files.
    """

    @staticmethod
    def load_system_config(config_file_path: str) -> Dict[str, Any]:
        """
        Loads the main system configuration from a specified YAML file.

        Args:
            config_file_path (str): The absolute or relative path to the
                                    YAML configuration file.

        Returns:
            Dict[str, Any]: A dictionary representing the loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If there's an error parsing the YAML file.
            ValueError: If the loaded config is not a dictionary or is empty.
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

        try:
            with open(config_file_path, 'r') as stream:
                config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML configuration file: {config_file_path}")
            # Log the exception details if a logger is available
            # For example: logger.error(f"YAML parsing error: {exc}")
            raise  # Re-raise the exception

        if not isinstance(config, dict):
            raise ValueError(f"Loaded configuration from {config_file_path} is not a dictionary.")
        if not config:
            raise ValueError(f"Loaded configuration from {config_file_path} is empty.")
            
        print(f"Configuration successfully loaded from: {config_file_path}")
        return config

    @staticmethod
    def validate_config(config: Dict[str, Any], expected_top_level_keys: list = None) -> bool:
        """
        Performs basic validation on the loaded configuration.
        Checks for the presence of expected top-level keys.

        Args:
            config (Dict[str, Any]): The loaded configuration dictionary.
            expected_top_level_keys (list, optional): A list of strings representing
                                                      the expected top-level keys.
                                                      If None, a default set is used.

        Returns:
            bool: True if basic validation passes, raises ValueError otherwise.
        
        Raises:
            ValueError: If required keys are missing.
        """
        if expected_top_level_keys is None:
            expected_top_level_keys = [
                "data_ingestion_preprocessing_config",
                "feature_engineering_config",
                "fhmv_core_engine_config",
                "signal_generation_config",
                "risk_management_config",
                "hpem_subtype_config", # Often part of risk or signal config, but listed separately in our example
                "backtesting_config",
                "portfolio_config"
            ]
        
        missing_keys = [key for key in expected_top_level_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Configuration validation failed. Missing top-level keys: {missing_keys}")
            
        # Further validation (e.g., type checking, value constraints) could be added here,
        # potentially using a library like Pydantic for more robust schema validation.
        print("Basic configuration validation passed.")
        return True

# Example Usage (typically called from main scripts in scripts/ directory):
# if __name__ == '__main__':
#     # Assuming a config file exists at ../configs/fhmv_system_config_default.yaml
#     # For this example, let's create a dummy one if it doesn't exist for testing
#     dummy_config_content = {
#         "data_ingestion_preprocessing_config": {"vol_atr_period": 10},
#         "feature_engineering_config": {},
#         "fhmv_core_engine_config": {"num_features": 10},
#         "signal_generation_config": {},
#         "risk_management_config": {},
#         "hpem_subtype_config": {},
#         "backtesting_config": {"initial_capital": 50000},
#         "portfolio_config": {}
#     }
#     example_config_path = "dummy_config.yaml"
#     if not os.path.exists(example_config_path):
#        try:
#            with open(example_config_path, 'w') as f:
#                yaml.dump(dummy_config_content, f)
#        except Exception as e:
#            print(f"Could not create dummy config for example: {e}")

#     try:
#         if os.path.exists(example_config_path):
#             system_config = ConfigLoader.load_system_config(example_config_path)
#             ConfigLoader.validate_config(system_config)
#             # print("\nLoaded Configuration:")
#             # print(yaml.dump(system_config, indent=2))
            
#             # Accessing a specific module's config:
#             # data_ingest_config = system_config.get("data_ingestion_preprocessing_config")
#             # if data_ingest_config:
#             #     print(f"\nATR Period from loaded config: {data_ingest_config.get('vol_atr_period')}")
#         else:
#             print(f"Skipping example usage as dummy config '{example_config_path}' could not be confirmed.")
#     except Exception as e:
#         print(f"Error in example usage: {e}")
#     finally:
#         if os.path.exists(example_config_path) and "dummy_config.yaml" in example_config_path : # Clean up dummy
#              os.remove(example_config_path)