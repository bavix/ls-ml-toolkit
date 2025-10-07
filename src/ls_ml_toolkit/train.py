#!/usr/bin/env python3
"""
Script for training YOLO model from Label Studio dataset
Downloads images from S3/HTTP, converts annotations to YOLO format, and trains the model
"""

import os
import sys
import json
import argparse
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError

# Import beautiful UI components
try:
    from .ui import (
        Banner, StatusDisplay, Table, 
        Colors, Icons, FileTree
    )
except ImportError:
    from ls_ml_toolkit.ui import (
        Banner, StatusDisplay, Table, 
        Colors, Icons, FileTree
    )

# Add src directory to path when running as script
# Get the directory containing this file
current_dir = Path(__file__).parent
# Add the parent directory (src) to sys.path
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Load environment variables from .env file for YAML substitution
try:
    from .env_loader import EnvLoader
    env = EnvLoader()
except ImportError:
    # Fallback if env_loader is not available
    class MockEnv:
        def get(self, key, default=None):
            return os.environ.get(key, default)
        def get_int(self, key, default=0):
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        def get_float(self, key, default=0.0):
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default
        def get_bool(self, key, default=False):
            value = os.environ.get(key, str(default)).lower()
            return value in ('true', '1', 'yes', 'on', 'enabled')
    env = MockEnv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelStudioToYOLOConverter:
    """Converter from Label Studio format to YOLO format"""
    
    def __init__(self, dataset_name: str, json_file: str, dataset_dir: str = "dataset", train_split: float = 0.8, val_split: float = 0.2, force_download: bool = False):
        self.dataset_name = dataset_name
        self.json_file = json_file
        self.dataset_dir = Path(dataset_dir) / dataset_name
        self.train_split = train_split
        self.val_split = val_split
        self.force_download = force_download
        
        # Create directory structure
        self.train_images_dir = self.dataset_dir / "train" / "images"
        self.train_labels_dir = self.dataset_dir / "train" / "labels"
        self.val_images_dir = self.dataset_dir / "val" / "images"
        self.val_labels_dir = self.dataset_dir / "val" / "labels"
        
        # Class mapping
        self.classes = ["price_major", "price_minor"]
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def validate_json(self) -> bool:
        """Validate Label Studio JSON format"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("JSON should contain a list of tasks")
                return False
            
            for i, task in enumerate(data):
                if not isinstance(task, dict):
                    logger.error(f"Task {i} is not a dictionary")
                    return False
                
                if 'data' not in task:
                    logger.error(f"Task {i} missing 'data' field")
                    return False
                
                if 'annotations' not in task:
                    logger.error(f"Task {i} missing 'annotations' field")
                    return False
                
                if 'image' not in task['data']:
                    logger.error(f"Task {i} missing 'image' field in data")
                    return False
            
            logger.info(f"✓ JSON validation passed. Found {len(data)} tasks")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating JSON: {e}")
            return False
    
    def create_directories(self):
        """Create directory structure for YOLO dataset"""
        try:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self.train_images_dir.mkdir(parents=True, exist_ok=True)
            self.train_labels_dir.mkdir(parents=True, exist_ok=True)
            self.val_images_dir.mkdir(parents=True, exist_ok=True)
            self.val_labels_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory structure: {e}")
            raise
    
    def create_yolo_config(self):
        """Create YOLO configuration files"""
        try:
            # Create classes.txt
            classes_file = self.dataset_dir / "classes.txt"
            with open(classes_file, 'w') as f:
                for cls in self.classes:
                    f.write(f"{cls}\n")
            
            # Create data.yaml
            data_yaml = self.dataset_dir / "data.yaml"
            yaml_content = f"""path: {self.dataset_dir.absolute()}
train: train/images
val: val/images

nc: {len(self.classes)}
names: {self.classes}
"""
            with open(data_yaml, 'w') as f:
                f.write(yaml_content)
            
        except Exception as e:
            logger.error(f"Error creating YOLO config files: {e}")
            raise
    
    def split_data(self):
        """Split data into train and validation sets"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sort data by task ID for consistent splitting
        data.sort(key=lambda x: x.get('id', 0))
        
        
        # Calculate split indices
        total_tasks = len(data)
        train_count = int(total_tasks * self.train_split)
        
        train_data = data[:train_count]
        val_data = data[train_count:]
        
        logger.info(f"✓ Split data: {len(train_data)} train ({self.train_split:.1%}), {len(val_data)} val ({self.val_split:.1%})")
        
        return train_data, val_data
    
    def download_image(self, url: str, output_path: Path) -> bool:
        """Download image from URL if it doesn't already exist or force_download is True"""
        try:
            # Check if file already exists and force_download is not set
            if output_path.exists() and not self.force_download:
                logger.debug(f"File already exists, skipping download: {output_path}")
                return True
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if url.startswith('s3://'):
                return self._download_from_s3(url, output_path)
            else:
                return self._download_from_http(url, output_path)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    def _download_from_http(self, url: str, output_path: Path) -> bool:
        """Download file from HTTP URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except requests.RequestException as e:
            logger.error(f"HTTP download error: {e}")
            return False
    
    def _download_from_s3(self, s3_url: str, output_path: Path) -> bool:
        """Download file from S3"""
        try:
            # Parse S3 URL
            parsed = urllib.parse.urlparse(s3_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # Get S3 credentials from config
            try:
                # Try relative import first (when used as module)
                from .config_loader import ConfigLoader
                config = ConfigLoader()
                s3_config = config.get_s3_config()
                s3_access_key = s3_config['access_key_id']
                s3_secret_key = s3_config['secret_access_key']
                s3_region = s3_config['region']
                s3_endpoint = s3_config['endpoint']
            except ImportError:
                # Try absolute import (when run as script)
                from ls_ml_toolkit.config_loader import ConfigLoader
                config = ConfigLoader()
                s3_config = config.get_s3_config()
                s3_access_key = s3_config['access_key_id']
                s3_secret_key = s3_config['secret_access_key']
                s3_region = s3_config['region']
                s3_endpoint = s3_config['endpoint']
            
            if not s3_access_key or not s3_secret_key:
                logger.error("S3 credentials not found in configuration")
                return False
            
            # Create S3 client
            s3_config = {
                'aws_access_key_id': s3_access_key,
                'aws_secret_access_key': s3_secret_key,
                'region_name': s3_region
            }
            
            if s3_endpoint:
                s3_config['endpoint_url'] = s3_endpoint
            
            s3_client = boto3.client('s3', **s3_config)
            
            # Download file
            s3_client.download_file(bucket, key, str(output_path))
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"S3 download error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            return False
    
    def process_dataset(self) -> bool:
        """Process the entire dataset"""
        try:
            # Validate JSON
            if not self.validate_json():
                return False
            
            # Create directories
            self.create_directories()
            
            # Create YOLO config
            self.create_yolo_config()
            
            # Split data
            train_data, val_data = self.split_data()
            
            # Process training data
            self._process_data_split(train_data, self.train_images_dir, self.train_labels_dir)
            self._process_data_split(val_data, self.val_images_dir, self.val_labels_dir)
            
            logger.info("✓ Dataset processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return False
    
    def _process_data_split(self, data: List[Dict], images_dir: Path, labels_dir: Path):
        """Process a data split (train or val)"""
        downloaded_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, task in enumerate(tqdm(data, desc="Processing tasks")):
            try:
                # Get image URL
                image_url = task['data']['image']
                
                # Create image filename
                image_filename = f"image_{task.get('id', i)}.jpg"
                image_path = images_dir / image_filename
                
                # Check if file already exists and force_download is not set
                if image_path.exists() and not self.force_download:
                    skipped_count += 1
                    logger.debug(f"Skipping download for existing file: {image_path}")
                else:
                    # Download image
                    if not self.download_image(image_url, image_path):
                        logger.warning(f"Failed to download {image_url}")
                        failed_count += 1
                        # Don't continue here - still process annotations even if download failed
                    else:
                        downloaded_count += 1
                
                # Process annotations (always, regardless of whether file was downloaded or skipped)
                annotations = task.get('annotations', [])
                if annotations:
                    # Get first annotation
                    annotation = annotations[0]
                    result = annotation.get('result', [])
                    
                    # Convert to YOLO format
                    yolo_annotations = []
                    for ann in result:
                        if ann.get('type') == 'rectanglelabels':
                            yolo_ann = self.convert_annotation_to_yolo(ann, image_path)
                            if yolo_ann:
                                yolo_annotations.append(yolo_ann)
                    
                    # Save labels
                    if yolo_annotations:
                        label_filename = f"image_{task.get('id', i)}.txt"
                        label_path = labels_dir / label_filename
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_annotations))
                
            except Exception as e:
                logger.error(f"Error processing task {i}: {e}")
                failed_count += 1
                continue
        
        # Log statistics
        total_processed = downloaded_count + skipped_count + failed_count
        logger.info(f"✓ Processed {total_processed} tasks: {downloaded_count} downloaded, {skipped_count} skipped, {failed_count} failed")
    
    def convert_annotation_to_yolo(self, annotation: Dict, image_path: Path) -> str:
        """Convert Label Studio annotation to YOLO format"""
        try:
            value = annotation['value']
            original_width = annotation.get('original_width', 1000)
            original_height = annotation.get('original_height', 1000)
            
            # Get class
            rectangle_labels = value.get('rectanglelabels', [])
            if not rectangle_labels:
                return None
            
            class_name = rectangle_labels[0]
            if class_name not in self.class_to_id:
                return None
            
            class_id = self.class_to_id[class_name]
            
            # Convert coordinates
            x = value['x'] / 100.0
            y = value['y'] / 100.0
            width = value['width'] / 100.0
            height = value['height'] / 100.0
            
            # Convert to YOLO format (center_x, center_y, width, height)
            center_x = x + width / 2
            center_y = y + height / 2
            
            return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            
        except Exception as e:
            logger.error(f"Error converting annotation: {e}")
            return None

class YOLOTrainer:
    """YOLO model trainer"""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
    
    def train_model(self, epochs: int = 50, imgsz: int = 640, batch: int = 8, device: str = "auto", nms_config: dict = None) -> bool:
        """Train YOLO model"""
        try:
            
            # Auto-detect device if needed
            if device == "auto":
                device = self._detect_best_device()
                logger.info(f"Auto-detected device: {device}")
            
            # Check if ultralytics is available
            try:
                import ultralytics
            except ImportError:
                logger.error("Ultralytics not found. Please install it first:")
                logger.error("  ml_env/bin/pip install ultralytics")
                return False
            
            # Create YOLO model
            from ultralytics import YOLO
            model = YOLO('yolo11n.pt')
            
            # Prepare training arguments
            train_args = {
                'data': str(self.dataset_dir / "data.yaml"),
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                'device': device,
                'project': "runs/detect",
                'name': "train",
                'save_period': 10,
                'patience': 50,
                'workers': 8
            }
            
            # Add NMS settings if provided
            if nms_config:
                train_args.update({
                    'iou': nms_config.get('iou_threshold', 0.7),
                    'conf': nms_config.get('conf_threshold', 0.25),
                    'max_det': nms_config.get('max_det', 300)
                })
                logger.info(f"NMS settings: iou={train_args['iou']}, conf={train_args['conf']}, max_det={train_args['max_det']}")
            
            # Train model
            results = model.train(**train_args)
            
            logger.info("✓ Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def _find_latest_training_dir(self) -> Path:
        """Find the latest training directory created by YOLO"""
        project_path = Path("runs/detect")
        if not project_path.exists():
            return None
        
        # Find all training directories
        training_dirs = []
        for item in project_path.iterdir():
            if item.is_dir() and item.name.startswith("train"):
                training_dirs.append(item)
        
        if not training_dirs:
            return None
        
        # Sort by modification time (newest first)
        training_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return training_dirs[0]
    
    def _detect_best_device(self) -> str:
        """Auto-detect the best available device for training"""
        try:
            import torch

            # Check for MPS (macOS Metal)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"

            # Check for CUDA (NVIDIA)
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                return "0"  # Use first GPU

            # Check for ROCm (AMD) - this is more complex to detect
            try:
                # Try to create a tensor on ROCm device
                test_tensor = torch.tensor([1.0]).to('cuda')
                logger.info("✓ ROCm (AMD) available")
                return "0"  # Use first GPU
            except Exception:
                pass

            # Fallback to CPU
            logger.info("✓ Using CPU for training")
            return "cpu"

        except ImportError:
            logger.info("✓ PyTorch not available, using CPU")
            return "cpu"
        except Exception as e:
            logger.warning(f"Error detecting device: {e}, falling back to CPU")
            return "cpu"

def main():
    # Display beautiful banner
    Banner.display()
    
    parser = argparse.ArgumentParser(
        description="Train YOLO model from Label Studio dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.BRIGHT_CYAN}Examples:{Colors.RESET}
  {Colors.DIM}# Basic training{Colors.RESET}
  lsml-train dataset.json --epochs 50 --batch 8

  {Colors.DIM}# Training with custom config{Colors.RESET}
  lsml-train dataset.json --config custom.yaml --device mps

  {Colors.DIM}# Force re-download images{Colors.RESET}
  lsml-train dataset.json --force-download --epochs 100
        """
    )
    
    parser.add_argument("json_file", help="Path to Label Studio JSON file")
    parser.add_argument("--dataset-name", help="Dataset name (default: basename of json file)")
    parser.add_argument("--dataset-dir", help="Dataset directory (default: from ls-ml-toolkit.yaml or 'dataset')")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (default: from ls-ml-toolkit.yaml or 50)")
    parser.add_argument("--imgsz", type=int, help="Image size for training (default: from ls-ml-toolkit.yaml or 640)")
    parser.add_argument("--batch", type=int, help="Batch size for training (default: from ls-ml-toolkit.yaml or 8)")
    parser.add_argument("--device", help="Device for training (default: from ls-ml-toolkit.yaml or auto)")
    parser.add_argument("--output-model", help="Output ONNX model path (default: from ls-ml-toolkit.yaml)")
    parser.add_argument("--config", help="Path to ls-ml-toolkit.yaml file (default: ls-ml-toolkit.yaml)")
    parser.add_argument("--train-split", type=float, help="Training data split ratio (default: from ls-ml-toolkit.yaml or 0.8)")
    parser.add_argument("--val-split", type=float, help="Validation data split ratio (default: from ls-ml-toolkit.yaml or 0.2)")
    parser.add_argument("--optimize", action='store_true', help="Enable model optimization after export")
    parser.add_argument("--no-optimize", action='store_true', help="Disable model optimization after export")
    parser.add_argument("--force-download", action='store_true', help="Force re-download of existing images")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    try:
        # Try relative import first (when used as module)
        from .config_loader import load_config
        config = load_config(args.config or "ls-ml-toolkit.yaml")
        config.apply_cli_args(args)
    except ImportError:
        # Try absolute import (when run as script)
        from ls_ml_toolkit.config_loader import load_config
        config = load_config(args.config or "ls-ml-toolkit.yaml")
        config.apply_cli_args(args)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure ls-ml-toolkit.yaml exists in the current directory")
        sys.exit(1)
    
    # Get configuration values from YAML
    epochs = args.epochs or config.get('training.epochs', 50)
    imgsz = args.imgsz or config.get('training.image_size', 640)
    batch = args.batch or config.get('training.batch_size', 8)
    device = args.device or config.get('training.device', 'auto')
    output_model = args.output_model or config.get('export.model_path', 'shared/models/layout_yolo_universal.onnx')
    dataset_dir = args.dataset_dir or config.get('dataset.base_dir', 'dataset')
    train_split = args.train_split or config.get('dataset.train_split', 0.8)
    val_split = args.val_split or config.get('dataset.val_split', 0.2)
    optimize = args.optimize or (not args.no_optimize and config.get('export.optimize', True))
    
    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = Path(args.json_file).stem
    
    # Display configuration in beautiful table
    logger.info(f"Starting training pipeline for dataset: {dataset_name}")
    
    config_table = Table(["Setting", "Value"], "modern")
    config_table.add_row(["JSON file", str(args.json_file)])
    config_table.add_row(["Epochs", str(epochs)])
    config_table.add_row(["Image size", str(imgsz)])
    config_table.add_row(["Batch size", str(batch)])
    config_table.add_row(["Device", str(device)])
    config_table.add_row(["Output model", str(output_model)])
    config_table.add_row(["Dataset directory", str(dataset_dir)])
    config_table.add_row(["Train split", f"{train_split:.1%}"])
    config_table.add_row(["Val split", f"{val_split:.1%}"])
    config_table.add_row(["Optimize model", "Yes" if optimize else "No"])
    config_table.add_row(["Force download", "Yes" if args.force_download else "No"])
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{Icons.CONFIG} Configuration{Colors.RESET}")
    config_table.display()
    
    try:
        # Create status display for the entire process
        status = StatusDisplay("Training Pipeline")
        status.add_step("Convert Label Studio dataset to YOLO format")
        status.add_step("Train YOLO model")
        status.add_step("Export to ONNX format")
        if optimize:
            status.add_step("Optimize ONNX model")
        status.start()
        
        # Step 1: Convert dataset
        status.update_step(0, "Processing dataset...")
        converter = LabelStudioToYOLOConverter(
            dataset_name, 
            args.json_file, 
            dataset_dir=dataset_dir,
            train_split=train_split,
            val_split=val_split,
            force_download=args.force_download
        )
        if not converter.process_dataset():
            logger.error("Dataset processing failed!")
            sys.exit(1)
        status.update_step(1, "Dataset conversion completed")
        
        # Step 2: Train model
        status.update_step(1, "Starting model training...")
        trainer = YOLOTrainer(converter.dataset_dir)
        
        # Get NMS settings from config
        nms_config = config.get("training", {}).get("nms")
        
        if not trainer.train_model(epochs, imgsz, batch, device, nms_config):
            logger.error("Model training failed!")
            sys.exit(1)
        status.update_step(2, "Model training completed")
        
        # Step 3: Export to ONNX
        status.update_step(2, "Exporting to ONNX format...")
        
        # Find the latest training directory created by YOLO
        latest_training_dir = trainer._find_latest_training_dir()
        if not latest_training_dir:
            logger.error("No training directory found!")
            sys.exit(1)
        
        trained_model = latest_training_dir / "weights" / "best.pt"
        logger.info(f"Using training results from: {latest_training_dir}")
        if trained_model.exists():
            try:
                from ultralytics import YOLO
                
                # Load trained model
                model = YOLO(str(trained_model))
                
                # Export to ONNX
                model.export(
                    format='onnx',
                    imgsz=640,
                    opset=11,
                    simplify=True
                )
                
                # Move exported model to target location
                onnx_file = trained_model.with_suffix('.onnx')
                if onnx_file.exists():
                    import shutil
                    shutil.move(str(onnx_file), output_model)
                    logger.info(f"✓ Model exported to {output_model}")
                else:
                    logger.error("ONNX export failed - no output file generated")
                    sys.exit(1)
                    
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")
                sys.exit(1)
        else:
            logger.error("Trained model not found!")
            sys.exit(1)
        
        # Step 4: Optimize ONNX model (if enabled)
        if optimize:
            status.update_step(3, "Optimizing ONNX model...")
            try:
                # Try relative import first (when used as module)
                from .optimize_onnx import optimize_onnx_model
            except ImportError:
                # Try absolute import (when run as script)
                from ls_ml_toolkit.optimize_onnx import optimize_onnx_model
            
            # Get optimized model path from config
            try:
                optimized_model_path = config.config["export"]["optimized_model_path"]
            except (KeyError, AttributeError):
                # Fallback to default naming if not specified in config
                optimized_model_path = str(Path(output_model).with_suffix('')) + '_optimized.onnx'
            
            try:
                if optimize_onnx_model(output_model, optimized_model_path, "all"):
                    logger.info(f"✓ Model optimization completed!")
                    logger.info(f"Optimized model: {optimized_model_path}")
                else:
                    logger.warning("Model optimization failed, but training completed successfully")
            except ImportError:
                logger.warning("ONNX optimization tools not available. Install with: pip install onnxruntime-tools")
            except Exception as e:
                logger.warning(f"Model optimization failed: {e}")
        else:
            logger.info("Model optimization skipped (disabled)")
        
        # Complete the process
        status.complete()
        
        # Display final results
        print(f"\n{Colors.BRIGHT_GREEN}{Icons.SUCCESS} Training pipeline completed successfully!{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}Final model: {Colors.BRIGHT_CYAN}{output_model}{Colors.RESET}")
        
        # Show file tree of results
        if latest_training_dir and latest_training_dir.exists():
            print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{Icons.FOLDER} Training Results {latest_training_dir}{Colors.RESET}")
            FileTree.display(latest_training_dir, max_depth=2)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
