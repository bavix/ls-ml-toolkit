#!/usr/bin/env python3
"""
Environment variables loader for Label Studio ML Toolkit
Loads environment variables from .env file with type-safe access
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class EnvLoader:
    """Load and manage environment variables from .env file"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.variables = {}
        self.load_env_file()
    
    def load_env_file(self):
        """Load environment variables from .env file"""
        if not self.env_file.exists():
            return
        
        with open(self.env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    self.variables[key] = value
                    # Also set in os.environ for compatibility with other modules
                    os.environ[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get environment variable as string"""
        return self.variables.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get environment variable as integer"""
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get environment variable as float"""
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean"""
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def list_variables(self) -> Dict[str, str]:
        """List all loaded variables (without sensitive values)"""
        sensitive_keys = {
            'LS_ML_S3_ACCESS_KEY_ID', 'LS_ML_S3_SECRET_ACCESS_KEY', 
            'PASSWORD', 'SECRET', 'KEY', 'TOKEN'
        }
        
        filtered_vars = {}
        for key, value in self.variables.items():
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                filtered_vars[key] = "***HIDDEN***"
            else:
                filtered_vars[key] = value
        
        return filtered_vars

def load_env(env_file: str = ".env") -> EnvLoader:
    """Load environment variables from file"""
    return EnvLoader(env_file)

def main():
    """Test the environment loader"""
    print("🔧 Environment Variables Loader Test")
    print("=" * 40)
    print()
    
    # Load environment variables
    env = load_env()
    
    # Show loaded variables
    print("📋 Loaded variables:")
    for key, value in env.list_variables().items():
        print(f"   {key}={value}")
    
    print()
    print("🧪 Testing getter methods:")
    
    # AWS Configuration (only S3 credentials in .env)
    print("📡 AWS S3 Credentials:")
    print(f"   LS_ML_AWS_ACCESS_KEY_ID: {env.get('LS_ML_AWS_ACCESS_KEY_ID', 'NOT_SET')}")
    print(f"   LS_ML_AWS_SECRET_ACCESS_KEY: {'SET' if env.get('LS_ML_AWS_SECRET_ACCESS_KEY') else 'NOT_SET'}")
    
    print("\n💡 Note: All other settings are configured in ls-ml-toolkit.yaml")
    print("   - Training parameters")
    print("   - Dataset configuration") 
    print("   - Platform settings")
    print("   - Development options")
    print("   - Advanced configuration")

if __name__ == "__main__":
    main()
