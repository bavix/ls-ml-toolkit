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
# Train a model from Label Studio dataset
lsml-train dataset/v0.json --epochs 50 --batch 8 --device auto

# Optimize an ONNX model
lsml-optimize model.onnx

# PyTorch automatically detects your platform and GPU
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

Create a `.env` file with your S3 credentials:

```bash
LS_ML_AWS_ACCESS_KEY_ID=your_access_key
LS_ML_AWS_SECRET_ACCESS_KEY=your_secret_key
```

### YAML Configuration (ls-ml-toolkit.yaml)

All other settings are configured in `ls-ml-toolkit.yaml`:

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

# AWS S3 Configuration
aws:
  access_key_id: "${LS_ML_AWS_ACCESS_KEY_ID}"
  secret_access_key: "${LS_ML_AWS_SECRET_ACCESS_KEY}"
  region: "us-east-1"
  endpoint: ""

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
lsml-train dataset/v0.json \
  --epochs 100 \
  --batch 16 \
  --device mps \
  --imgsz 640 \
  --optimize
```

### Using Configuration File

```bash
lsml-train dataset/v0.json --config custom-config.yaml
```

## File Structure

```
ls-ml-toolkit/
├── ls_ml_toolkit/              # Main package source
│   ├── __init__.py
│   ├── train.py                # Main training script
│   ├── config_loader.py        # Configuration management
│   ├── env_loader.py           # Environment variable loader
│   ├── optimize_onnx.py        # ONNX optimization
│   ├── install_deps.py         # Dependency installer
│   └── setup_env.py            # Environment setup
├── requirements.txt            # Dependencies
├── pyproject.toml             # Package configuration
├── setup.py                   # Setup script
├── ls-ml-toolkit.yaml         # Main configuration
├── env.example                # Environment template
└── README.md                  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
