# Label Studio ML Toolkit Makefile
# Simple commands for training and testing

.PHONY: help setup test train clean

# Default target
help:
	@echo "🚀 Label Studio ML Toolkit"
	@echo "========================="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup     - Set up virtual environment and install dependencies"
	@echo "  make test      - Run installation tests"
	@echo "  make train     - Train model with test dataset"
	@echo "  make clean     - Clean up generated files"
	@echo ""

# Set up virtual environment and install dependencies
setup:
	@echo "🔧 Setting up virtual environment..."
	python3 -m venv ml_env
	ml_env/bin/pip install --upgrade pip
	ml_env/bin/pip install -r requirements.txt
	@echo "✅ Setup complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	ml_env/bin/python simple_test.py

# Train model with test dataset
train:
	@echo "🚀 Training model..."
	ml_env/bin/python train.py test_dataset.json --epochs 1 --batch 1 --device cpu

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf ml_env/
	rm -rf dataset/
	rm -rf runs/
	rm -rf shared/
	@echo "✅ Cleanup complete!"
