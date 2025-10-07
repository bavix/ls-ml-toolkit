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
- **Modern CLI Interface**: Beautiful terminal output with progress indicators and status displays
- **Smart NMS Configuration**: Optimized Non-Maximum Suppression settings to reduce warnings
- **Automatic Training Directory Detection**: Finds the latest YOLO training output automatically

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
  
  # NMS (Non-Maximum Suppression) settings
  nms:
    iou_threshold: 0.7  # IoU threshold for NMS (0.0-1.0) - higher = fewer detections
    conf_threshold: 0.25  # Confidence threshold for predictions (0.0-1.0) - higher = fewer detections
    max_det: 300  # Maximum number of detections per image - lower = faster processing

# Model Export Configuration
export:
  model_path: "shared/models/layout_yolo_universal.onnx"
  optimized_model_path: "shared/models/layout_yolo_universal_optimized.onnx"  # Optional
  optimize: true
  optimization_level: "all"

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
# Build package
python -m build

# Install in development mode
pip install -e .
```

## Command Line Tools

- **`lsml-train`**: Train YOLO models from Label Studio datasets
- **`lsml-optimize`**: Optimize ONNX models for deployment

### CLI Features

- **Beautiful Interface**: Modern terminal UI with colors, icons, and progress indicators
- **Status Tracking**: Real-time progress updates during training and optimization
- **Configuration Display**: Shows current settings in a formatted table
- **File Tree Display**: Visual representation of training results and file structure
- **Error Handling**: Clear error messages and troubleshooting guidance

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
  --optimize \
  --force-download
```

### Using Configuration File

```bash
# Use custom YAML configuration
lsml-train dataset/v0.json --config custom-config.yaml

# Override specific settings via command line
lsml-train dataset/v0.json --epochs 100 --batch 16 --device mps
```

### Advanced Usage Examples

```bash
# Force re-download of existing images
lsml-train dataset/v0.json --force-download

# Train with custom NMS settings (via YAML config)
# Edit ls-ml-toolkit.yaml:
# training:
#   nms:
#     iou_threshold: 0.8
#     conf_threshold: 0.3
#     max_det: 200

# Optimize existing ONNX model
lsml-optimize model.onnx --level extended

# Use custom output path for optimization
lsml-optimize model.onnx --output optimized_model.onnx
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
# S3 Configuration (uses .env variables)
s3:
  access_key_id: "${LS_ML_S3_ACCESS_KEY_ID}"  # From .env file
  secret_access_key: "${LS_ML_S3_SECRET_ACCESS_KEY}"  # From .env file
  region: "${LS_ML_S3_DEFAULT_REGION:-us-east-1}"  # From .env with default
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

## Model Export Configuration

### Model Paths
- **`model_path`**: Path for the regular ONNX export (required)
- **`optimized_model_path`**: Path for the optimized ONNX model (optional)

### Fallback Behavior
If `optimized_model_path` is not specified in the configuration:
- **Training script**: Uses `{model_path}_optimized.onnx` as fallback
- **Optimization script**: Uses `{input_model}_optimized.onnx` as fallback

### Examples
```yaml
export:
  model_path: "models/my_model.onnx"
  optimized_model_path: "models/my_model_optimized.onnx"  # Optional
  optimize: true
  optimization_level: "all"
```
- **All non-sensitive settings**

### 🔒 Security:
- Never commit `.env` files to version control
- Use `.env.example` as a template
- Keep sensitive data separate from code

## File Structure

```
ls-ml-toolkit/
├── src/
│   └── ls_ml_toolkit/         # Main package source
│       ├── __init__.py
│       ├── train.py            # Main training script
│       ├── config_loader.py    # Configuration management with .env support
│       ├── env_loader.py       # Environment variable loader
│       ├── optimize_onnx.py    # ONNX optimization
│       └── ui.py               # CLI UI components
├── tests/                      # Test files
├── requirements.txt            # Dependencies
├── pyproject.toml             # Package configuration
├── setup.py                   # Setup script
├── ls-ml-toolkit.yaml         # Main configuration with env var substitution
├── env.example                # Environment template
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

### NMS Time Limit Warnings

If you see `WARNING ⚠️ NMS time limit 2.800s exceeded`:

**What it means:**
- NMS (Non-Maximum Suppression) operation is taking too long
- This can slow down validation and inference
- Usually happens with many objects or suboptimal settings

**How to fix:**
1. **Optimize NMS settings** in `ls-ml-toolkit.yaml`:
   ```yaml
   training:
     nms:
       iou_threshold: 0.8    # Higher = fewer detections (0.7-0.9)
       conf_threshold: 0.3   # Higher = fewer detections (0.25-0.5)
       max_det: 200          # Lower = fewer detections (100-300)
   ```

2. **Reduce batch size** if memory allows:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8 to 4
   ```

3. **Optimize other parameters**: Focus on `iou_threshold`, `conf_threshold`, and `max_det` for better performance

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

### Training Directory Issues

If the script can't find the latest training directory:

1. **Check YOLO output**: Ensure `runs/detect/` directory exists
2. **Verify permissions**: Make sure the script can read the directory
3. **Manual path**: The script automatically finds the latest `train*` directory

### ONNX Optimization Issues

If ONNX optimization fails:

1. **Install dependencies**: `pip install onnx onnxruntime`
2. **Check model format**: Ensure input is a valid ONNX model
3. **Use fallback**: The script will use default naming if config path is missing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
