#!/usr/bin/env python3
"""
ONNX Model Optimizer
Optimizes ONNX models for mobile deployment using onnxruntime-tools
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path when running as script
current_dir = Path(__file__).parent
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optimize_onnx_model(input_path: str, output_path: str, optimization_level: str = "all") -> bool:
    """
    Simple ONNX model optimization using basic techniques
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to output optimized ONNX model
        optimization_level: Optimization level (basic, extended, all)
    
    Returns:
        bool: True if optimization successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        import onnx
        import shutil
        
        logger.info(f"Loading ONNX model from {input_path}")
        
        # Load the model
        model = onnx.load(input_path)
        
        logger.info(f"Applying basic optimizations (level: {optimization_level})")
        
        # Basic optimizations that don't require external tools
        # 1. Remove unused initializers
        if optimization_level in ["extended", "all"]:
            used_initializers = set()
            for node in model.graph.node:
                for input_name in node.input:
                    used_initializers.add(input_name)
            
            # Remove unused initializers
            initializers_to_remove = []
            for initializer in model.graph.initializer:
                if initializer.name not in used_initializers:
                    initializers_to_remove.append(initializer)
            
            for initializer in initializers_to_remove:
                model.graph.initializer.remove(initializer)
                logger.info(f"Removed unused initializer: {initializer.name}")
        
        # 2. Basic graph cleanup
        if optimization_level in ["extended", "all"]:
            # Remove duplicate nodes (basic check)
            seen_nodes = set()
            nodes_to_remove = []
            
            for i, node in enumerate(model.graph.node):
                node_key = (node.op_type, tuple(node.input), tuple(node.output))
                if node_key in seen_nodes:
                    nodes_to_remove.append(i)
                else:
                    seen_nodes.add(node_key)
            
            # Remove duplicate nodes (in reverse order to maintain indices)
            for i in reversed(nodes_to_remove):
                model.graph.node.remove(model.graph.node[i])
                logger.info(f"Removed duplicate node at index {i}")
        
        # 3. For now, just copy the model with basic cleanup
        # In a real implementation, you would use onnx-simplifier or onnx-optimizer
        logger.info("Applying basic model cleanup...")
        
        # Save the model
        logger.info(f"Saving optimized model to {output_path}")
        onnx.save(model, output_path)
        
        # Get file sizes for comparison
        input_size = Path(input_path).stat().st_size
        output_size = Path(output_path).stat().st_size
        reduction = ((input_size - output_size) / input_size) * 100 if input_size > 0 else 0
        
        logger.info(f"✓ Model optimization completed!")
        logger.info(f"  Input size: {input_size / (1024*1024):.2f} MB")
        logger.info(f"  Output size: {output_size / (1024*1024):.2f} MB")
        logger.info(f"  Size reduction: {reduction:.1f}%")
        
        if reduction < 1.0:
            logger.info("  Note: For better optimization, install onnx-simplifier:")
            logger.info("        pip install onnx-simplifier")
        
        return True
        
    except ImportError as e:
        logger.error(f"Required packages not found: {e}")
        logger.error("Please install: pip install onnx")
        return False
    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        return False

def main():
    # Display beautiful banner
    Banner.display()
    
    parser = argparse.ArgumentParser(
        description="Optimize ONNX model for mobile deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.BRIGHT_CYAN}Examples:{Colors.RESET}
  {Colors.DIM}# Basic optimization{Colors.RESET}
  lsml-optimize model.onnx

  {Colors.DIM}# Custom output path{Colors.RESET}
  lsml-optimize model.onnx --output optimized_model.onnx

  {Colors.DIM}# Extended optimization{Colors.RESET}
  lsml-optimize model.onnx --level extended
        """
    )
    
    parser.add_argument("input_model", help="Path to input ONNX model")
    parser.add_argument("--output", "-o", help="Path to output optimized model (default: input_model_optimized.onnx)")
    parser.add_argument("--level", "-l", choices=["basic", "extended", "all"], 
                       help="Optimization level (default: from ls-ml-toolkit.yaml)")
    parser.add_argument("--config", help="Path to ls-ml-toolkit.yaml file (default: ls-ml-toolkit.yaml)")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    try:
        # Try relative import first (when used as module)
        from .config_loader import load_config
        config = load_config(args.config or "ls-ml-toolkit.yaml")
    except ImportError:
        # Try absolute import (when run as script)
        from ls_ml_toolkit.config_loader import load_config
        config = load_config(args.config or "ls-ml-toolkit.yaml")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please ensure ls-ml-toolkit.yaml exists in the current directory")
        sys.exit(1)
    
    # Get configuration values from YAML
    optimization_level = args.level or config.get('export.optimization_level', 'all')
    
    input_path = Path(args.input_model)
    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Try to get default output path from config
        try:
            config = config_loader.load_config()
            output_path = Path(config["export"]["optimized_model_path"])
        except (KeyError, FileNotFoundError):
            # Fallback to default naming
            output_path = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Display configuration in beautiful table
    config_table = Table(["Setting", "Value"], "modern")
    config_table.add_row(["Input model", str(input_path)])
    config_table.add_row(["Output model", str(output_path)])
    config_table.add_row(["Optimization level", optimization_level])
    config_table.add_row(["Model size", f"{input_path.stat().st_size / 1024 / 1024:.1f} MB"])
    
    print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{Icons.OPTIMIZE} Optimization Configuration{Colors.RESET}")
    config_table.display()
    
    # Create status display
    status = StatusDisplay("ONNX Model Optimization")
    status.add_step("Load ONNX model")
    status.add_step("Apply optimizations")
    status.add_step("Save optimized model")
    status.start()
    
    # Optimize model
    status.update_step(0, "Loading model...")
    if optimize_onnx_model(str(input_path), str(output_path), optimization_level):
        status.update_step(1, "Optimizations applied")
        status.update_step(2, "Model saved")
        status.complete()
        
        # Display results
        print(f"\n{Colors.BRIGHT_GREEN}{Icons.SUCCESS} Optimization completed successfully!{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}Optimized model: {Colors.BRIGHT_CYAN}{output_path}{Colors.RESET}")
        
        # Show file size comparison
        original_size = input_path.stat().st_size / 1024 / 1024
        optimized_size = output_path.stat().st_size / 1024 / 1024
        reduction = (1 - optimized_size / original_size) * 100
        
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_WHITE}{Icons.FILE} File Size Comparison{Colors.RESET}")
        size_table = Table(["Model", "Size"], "modern")
        size_table.add_row(["Original", f"{original_size:.1f} MB"])
        size_table.add_row(["Optimized", f"{optimized_size:.1f} MB"])
        size_table.add_row(["Reduction", f"{reduction:.1f}%"])
        size_table.display()
        
    else:
        logger.error("Optimization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
