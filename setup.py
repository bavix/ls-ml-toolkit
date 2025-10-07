#!/usr/bin/env python3
"""
Setup script for ls-ml-toolkit
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "Label Studio ML Toolkit"

# No platform-specific extras needed - PyTorch auto-detects GPU support

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "ultralytics>=8.0.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "PyYAML>=6.0.0",
        "boto3>=1.34.0",
        "botocore>=1.34.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ]

setup(
    name="ls-ml-toolkit",
    version="1.0.1",
    author="Babichev Maxim",
    author_email="info@babichev.net",
    description="Label Studio ML Toolkit: Convert, Train, Optimize object detection models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bavix/ls-ml-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
            entry_points={
                "console_scripts": [
                    "lsml-train=ls_ml_toolkit.train:main",
                    "lsml-optimize=ls_ml_toolkit.optimize_onnx:main",
                ],
            },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
    },
    keywords=[
        "label-studio",
        "yolo",
        "object-detection",
        "machine-learning",
        "computer-vision",
        "ml-toolkit",
        "dataset-conversion",
        "model-training",
        "onnx-optimization",
    ],
    project_urls={
        "Bug Reports": "https://github.com/bavix/ls-ml-toolkit/issues",
        "Source": "https://github.com/bavix/ls-ml-toolkit",
        "Documentation": "https://github.com/bavix/ls-ml-toolkit#readme",
    },
)
