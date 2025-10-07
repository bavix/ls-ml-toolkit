#!/usr/bin/env python3
"""
Configuration loader for Label Studio ML Toolkit
Loads configuration from YAML file with command-line argument overrides
"""

import os
import yaml
import argparse
import re
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_file: str = "ls-ml-toolkit.yaml"):
        self.config_file = Path(config_file)
        self.config = {}
        self._load_env_file()
        self.load_config()
    
    def _load_env_file(self):
        """Load .env file if it exists"""
        try:
            from .env_loader import EnvLoader
            env_loader = EnvLoader()
            # EnvLoader already loads variables into os.environ
        except ImportError:
            # Fallback: try to load .env file manually
            env_file = Path('.env')
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
            
            # Process environment variables in the config
            self.config = self._process_env_variables(raw_config)
            return self.config
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {self.config_file}: {e}")
    
    def _process_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variables in configuration"""
        if isinstance(config, dict):
            return {key: self._process_env_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._process_env_variables(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config
    
    def _substitute_env_vars(self, value: str) -> str:
        """Substitute environment variables in string values"""
        if not isinstance(value, str):
            return value
        
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            # Get environment variable value
            env_value = os.environ.get(var_name)
            
            # If env var exists, use it; otherwise use default (even if empty)
            if env_value is not None:
                return env_value
            else:
                # Use default value (even if it's empty)
                return default_value
        
        return re.sub(pattern, replace_var, value)
    
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def apply_cli_args(self, args: argparse.Namespace):
        """Apply command-line arguments to configuration"""
        # Map CLI arguments to config keys
        cli_mapping = {
            'dataset_dir': 'dataset.base_dir',
            'epochs': 'training.epochs',
            'imgsz': 'training.image_size',
            'batch': 'training.batch_size',
            'device': 'training.device',
            'output_model': 'export.model_path',
            'train_split': 'dataset.train_split',
            'val_split': 'dataset.val_split',
            'optimize': 'export.optimize'
        }
        
        for cli_key, config_key in cli_mapping.items():
            cli_value = getattr(args, cli_key, None)
            if cli_value is not None:
                self._set_nested_key(config_key, cli_value)
    
    def _set_nested_key(self, key: str, value: Any):
        """Set nested configuration key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return {
            'base_dir': self.get('dataset.base_dir', 'dataset'),
            'train_split': self.get('dataset.train_split', 0.8),
            'val_split': self.get('dataset.val_split', 0.2),
            'structure': self.get('dataset.structure', {})
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'model': self.get('training.model', 'yolo11n.pt'),
            'epochs': self.get('training.epochs', 50),
            'batch_size': self.get('training.batch_size', 8),
            'image_size': self.get('training.image_size', 640),
            'device': self.get('training.device', 'auto'),
            'project_dir': self.get('training.project_dir', 'runs/detect'),
            'name': self.get('training.name', 'train'),
            'save_period': self.get('training.save_period', 10),
            'patience': self.get('training.patience', 50),
            'workers': self.get('training.workers', 8)
        }
    
    def get_export_config(self) -> Dict[str, Any]:
        """Get export configuration"""
        return {
            'model_path': self.get('export.model_path', 'shared/models/layout_yolo_universal.onnx'),
            'format': self.get('export.format', 'onnx'),
            'imgsz': self.get('export.imgsz', 640),
            'opset': self.get('export.opset', 11),
            'simplify': self.get('export.simplify', True),
            'optimize': self.get('export.optimize', True),
            'optimization_level': self.get('export.optimization_level', 'all')
        }
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration"""
        return {
            'access_key_id': self.get('s3.access_key_id', ''),
            'secret_access_key': self.get('s3.secret_access_key', ''),
            'region': self.get('s3.region', 'us-east-1'),
            'endpoint': self.get('s3.endpoint', '')
        }
    
    def get_development_config(self) -> Dict[str, Any]:
        """Get development configuration"""
        return {
            'debug': self.get('development.debug', False),
            'test_mode': self.get('development.test_mode', False),
            'log_level': self.get('development.log_level', 'INFO')
        }
    
    def get_platform_config(self) -> Dict[str, Any]:
        """Get platform configuration"""
        return {
            'auto_detect_gpu': self.get('platform.auto_detect_gpu', True),
            'force_device': self.get('platform.force_device'),
            'macos': self.get('platform.macos', {}),
            'linux_nvidia': self.get('platform.linux_nvidia', {}),
            'linux_amd': self.get('platform.linux_amd', {})
        }
    
    
    def print_config(self):
        """Print current configuration"""
        print("📋 Current Configuration:")
        print("=" * 40)
        
        print("📁 Dataset:")
        print(f"  Base directory: {self.get('dataset.base_dir', 'dataset')}")
        print(f"  Train split: {self.get('dataset.train_split', 0.8)}")
        print(f"  Val split: {self.get('dataset.val_split', 0.2)}")
        
        print("\n🚀 Training:")
        print(f"  Model: {self.get('training.model', 'yolo11n.pt')}")
        print(f"  Epochs: {self.get('training.epochs', 50)}")
        print(f"  Batch size: {self.get('training.batch_size', 8)}")
        print(f"  Image size: {self.get('training.image_size', 640)}")
        print(f"  Device: {self.get('training.device', 'auto')}")
        
        print("\n📱 Export:")
        print(f"  Output model: {self.get('export.model_path', 'shared/models/layout_yolo_universal.onnx')}")
        print(f"  Format: {self.get('export.format', 'onnx')}")
        print(f"  Optimize: {self.get('export.optimize', True)}")

def load_config(config_file: str = "ls-ml-toolkit.yaml") -> ConfigLoader:
    """Load configuration from file"""
    return ConfigLoader(config_file)

def main():
    """Test the configuration loader"""
    print("🔧 Configuration Loader Test")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Print configuration
    config.print_config()
    
    print("\n🧪 Testing getter methods:")
    print(f"  Dataset base dir: {config.get('dataset.base_dir')}")
    print(f"  Training epochs: {config.get('training.epochs')}")
    print(f"  Export model path: {config.get('export.model_path')}")

if __name__ == "__main__":
    main()
