# LS-ML-Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ls-ml-toolkit.svg)](https://badge.fury.io/py/ls-ml-toolkit)

A comprehensive machine learning toolkit for converting Label Studio annotations, training object detection models, and optimizing for deployment.

## Features

- **Label Studio to YOLO Conversion**: Convert Label Studio JSON exports to YOLO format
- **Image Downloading**: Download images from S3/HTTP sources with progress tracking
- **YOLO Model Training**: Train YOLOv11 models with automatic device detection
- **ONNX Export & Optimization**: Export and optimize models for mobile deployment
- **Cross-Platform GPU Support**: MPS (macOS), CUDA (NVIDIA), ROCm (AMD)
- **Centralized Configuration**: YAML-based configuration with environment variable support
- **Automatic .env Loading**: Seamless integration with .env files for sensitive credentials
- **Environment Variable Substitution**: Support for `${VAR_NAME}` and `${VAR_NAME:-default}` syntax in YAML
- **Flexible Import System**: Works both as a Python module and as standalone scripts
- **Secure Configuration**: Sensitive data in .env, regular settings in YAML

## Quick Start

### Installation

```bash
# Install package (includes GPU support for all platforms)
pip install ls-ml-toolkit

# PyTorch automatically detects and uses:
# - macOS: Metal Performance Shaders (MPS)
# - Linux: CUDA/ROCm (if available)
# - Windows: CUDA (if available)
```

### Basic Usage

```bash
# 1. Create .env file with your S3 credentials
cp env.example .env
# Edit .env with your AWS credentials

# 2. Train a model from Label Studio dataset
lsml-train dataset/v0.json --epochs 50 --batch 8 --device auto

# 3. Optimize an ONNX model
lsml-optimize model.onnx

# PyTorch automatically detects your platform and GPU
# All configuration is loaded automatically from .env and ls-ml-toolkit.yaml
```

### Python API

```python
from ls_ml_toolkit import LabelStudioToYOLOConverter, YOLOTrainer

# Convert dataset
converter = LabelStudioToYOLOConverter('dataset_name', 'path/to/labelstudio.json')
converter.process_dataset()

# Train model
trainer = YOLOTrainer('path/to/dataset')
trainer.train_model(epochs=50, device='auto')
```

## Configuration

### Environment Variables (.env)

Create a `.env` file with your sensitive credentials only:

```bash
# S3 Credentials (Sensitive Data)
LS_ML_S3_ACCESS_KEY_ID=your_access_key
LS_ML_S3_SECRET_ACCESS_KEY=your_secret_key

# Optional: Environment-specific settings
LS_ML_S3_DEFAULT_REGION=us-east-1
LS_ML_S3_ENDPOINT=https://custom-s3.example.com
```

**Important**: 
- Only use `.env` for **sensitive data** (API keys, passwords, tokens)
- All other configuration should be in `ls-ml-toolkit.yaml`
- Copy `env.example` to `.env` and configure your credentials
- The toolkit automatically loads these variables and makes them available throughout the application

### YAML Configuration (ls-ml-toolkit.yaml)

All regular settings are configured in `ls-ml-toolkit.yaml`. Environment variables are used only for sensitive data:

```yaml
# Dataset Configuration
dataset:
  base_dir: "dataset"
  train_split: 0.8
  val_split: 0.2

# Training Configuration
training:
  epochs: 50
  batch_size: 8
  image_size: 640
  device: "auto"

# S3 Configuration (uses .env for sensitive data)
s3:
  access_key_id: "${LS_ML_S3_ACCESS_KEY_ID}"  # From .env file
  secret_access_key: "${LS_ML_S3_SECRET_ACCESS_KEY}"  # From .env file
  region: "${LS_ML_S3_DEFAULT_REGION:-us-east-1}"  # From .env file with default
  endpoint: "${LS_ML_S3_ENDPOINT:-}"  # From .env file (optional)

# Platform-specific settings
platform:
  auto_detect_gpu: true
  force_device: null
  macos:
    device: "mps"
    batch_size: 16
  linux:
    device: "auto"  # PyTorch will auto-detect GPU
    batch_size: 16
```

## Platform Support

### macOS
- **MPS Support**: Automatic Metal Performance Shaders detection
- **Installation**: `pip install ls-ml-toolkit`

### Linux
- **CUDA Support**: Automatic NVIDIA GPU detection and configuration
- **ROCm Support**: Automatic AMD GPU detection
- **Installation**: `pip install ls-ml-toolkit`
- **Requirements**: NVIDIA drivers + CUDA toolkit OR ROCm drivers

### Windows
- **CUDA Support**: Automatic NVIDIA GPU detection
- **Installation**: `pip install ls-ml-toolkit`
- **Requirements**: NVIDIA drivers + CUDA toolkit

## Development

### Setup Development Environment

```bash
git clone https://github.com/bavix/ls-ml-toolkit.git
cd ls-ml-toolkit
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest tests/
```

### Building Packages

```bash
# Build all platform-specific packages
python build_all.py

# Build individual packages
python -m build
```

## Command Line Tools

- **`lsml-train`**: Train YOLO models from Label Studio datasets
- **`lsml-optimize`**: Optimize ONNX models for deployment

## Examples

### Training with Custom Configuration

```bash
# Method 1: Use .env file (recommended for secrets)
echo "LS_ML_S3_ACCESS_KEY_ID=your_key" >> .env
echo "LS_ML_S3_SECRET_ACCESS_KEY=your_secret" >> .env

# Method 2: Use environment variables
export LS_ML_S3_ACCESS_KEY_ID="your_key"
export LS_ML_S3_SECRET_ACCESS_KEY="your_secret"

# Train with custom settings
lsml-train dataset/v0.json \
  --epochs 100 \
  --batch 16 \
  --device mps \
  --imgsz 640 \
  --optimize
```

### Using Configuration File

```bash
# Use custom YAML configuration
lsml-train dataset/v0.json --config custom-config.yaml

# Override specific settings via command line
lsml-train dataset/v0.json --epochs 100 --batch 16 --device mps
```

### Quick Setup Guide

```bash
# 1. Clone and install
git clone https://github.com/bavix/ls-ml-toolkit.git
cd ls-ml-toolkit
pip install -e .

# 2. Setup credentials
cp env.example .env
# Edit .env with your AWS credentials

# 3. Train your model
lsml-train your_dataset.json --epochs 50 --batch 8
```

### Environment Variable Substitution

The YAML configuration supports environment variable substitution **only for sensitive data**:

```yaml
# AWS S3 Configuration (uses .env variables)
aws:
  access_key_id: "${LS_ML_AWS_ACCESS_KEY_ID}"  # From .env file
  secret_access_key: "${LS_ML_AWS_SECRET_ACCESS_KEY}"  # From .env file
  region: "${LS_ML_AWS_DEFAULT_REGION:-us-east-1}"  # From .env with default
  endpoint: "${LS_ML_S3_ENDPOINT:-}"  # From .env (optional)

# Regular configuration (no env vars needed)
training:
  epochs: 50
  batch_size: 8
  image_size: 640
```

**Naming Convention**: `LS_ML_<CATEGORY>_<SETTING>`
- `LS_ML_S3_ACCESS_KEY_ID` - S3 credentials
- `LS_ML_S3_SECRET_ACCESS_KEY` - S3 credentials  
- `LS_ML_S3_DEFAULT_REGION` - S3 configuration
- `LS_ML_S3_ENDPOINT` - S3 endpoint

## Configuration Best Practices

### ✅ Use .env for:
- **API Keys & Secrets**: `LS_ML_S3_ACCESS_KEY_ID`, `LS_ML_S3_SECRET_ACCESS_KEY`
- **Environment-specific settings**: `LS_ML_S3_DEFAULT_REGION`, `LS_ML_S3_ENDPOINT`
- **Values that change between deployments**

### ✅ Use YAML for:
- **Regular configuration**: epochs, batch_size, image_size
- **Default values**: model paths, directory structures
- **Platform settings**: device detection, optimization levels
- **All non-sensitive settings**

### 🔒 Security:
- Never commit `.env` files to version control
- Use `.env.example` as a template
- Keep sensitive data separate from code

## File Structure

```
ls-ml-toolkit/
├── ls_ml_toolkit/              # Main package source
│   ├── __init__.py
│   ├── train.py                # Main training script
│   ├── config_loader.py        # Configuration management with .env support
│   ├── env_loader.py           # Environment variable loader
│   ├── optimize_onnx.py        # ONNX optimization
│   ├── install_deps.py         # Dependency installer
│   └── setup_env.py            # Environment setup
├── requirements.txt            # Dependencies
├── pyproject.toml             # Package configuration
├── setup.py                   # Setup script
├── ls-ml-toolkit.yaml         # Main configuration with env var substitution
├── .env.example               # Environment template
├── .env                       # Your environment variables (create from .env.example)
└── README.md                  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Troubleshooting

### Environment Variables Not Loading

If your `.env` file is not being loaded:

1. **Check file location**: Ensure `.env` is in the project root directory
2. **Verify file format**: Use `KEY=value` format (no spaces around `=`)
3. **Check permissions**: Ensure the file is readable
4. **Copy from template**: Use `cp env.example .env` as a starting point
5. **Check naming**: Use exact variable names like `LS_ML_S3_ACCESS_KEY_ID`

### YAML Variable Substitution Issues

If environment variables are not substituted in YAML:

1. **Check variable names**: Use exact names like `LS_ML_S3_ACCESS_KEY_ID`
2. **Verify syntax**: Use `${VAR_NAME}` or `${VAR_NAME:-default}` format
3. **Test loading**: Run `python -c "from ls_ml_toolkit.config_loader import ConfigLoader; print(ConfigLoader().get_s3_config())"`
4. **Remember**: Only use env vars for sensitive data, not regular config

### Import Errors

If you get import errors when running scripts:

1. **Install in development mode**: `pip install -e .`
2. **Check Python path**: Ensure the package is in your Python path
3. **Use absolute imports**: The toolkit supports both relative and absolute imports

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
