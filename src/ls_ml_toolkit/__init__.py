"""
Label Studio ML Toolkit

A comprehensive machine learning toolkit for converting Label Studio
annotations, training object detection models, and optimizing for deployment.

This package provides:
- Label Studio to YOLO dataset conversion
- Image downloading from S3/HTTP sources
- YOLO model training with automatic device detection
- ONNX model export and optimization
- Cross-platform GPU support (MPS, CUDA, ROCm)

Usage:
    from ls_ml_toolkit import LabelStudioToYOLOConverter, YOLOTrainer
    
    # Convert dataset
    converter = LabelStudioToYOLOConverter('dataset_name', 'path/to/labelstudio.json')
    converter.process_dataset()
    
    # Train model
    trainer = YOLOTrainer('path/to/dataset')
    trainer.train_model(epochs=50, device='auto')
"""

__version__ = "1.0.2"
__author__ = "Babichev Maxim"
__email__ = "info@babichev.net"

# Import main classes for easy access
try:
    from .train import LabelStudioToYOLOConverter, YOLOTrainer
    from .optimize_onnx import optimize_onnx_model
    from .env_loader import EnvLoader
except ImportError:
    # Handle import errors gracefully
    LabelStudioToYOLOConverter = None
    YOLOTrainer = None
    optimize_onnx_model = None
    EnvLoader = None

__all__ = [
    "LabelStudioToYOLOConverter",
    "YOLOTrainer", 
    "optimize_onnx_model",
    "EnvLoader",
]
