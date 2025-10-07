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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        reduction = ((input_size - output_size) / input_size) * 100
        
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
    parser = argparse.ArgumentParser(description="Optimize ONNX model for mobile deployment")
    parser.add_argument("input_model", help="Path to input ONNX model")
    parser.add_argument("--output", "-o", help="Path to output optimized model (default: input_model_optimized.onnx)")
    parser.add_argument("--level", "-l", choices=["basic", "extended", "all"], default="all",
                       help="Optimization level (default: all)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_model)
    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_optimized{input_path.suffix}"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optimize model
    if optimize_onnx_model(str(input_path), str(output_path), args.level):
        logger.info(f"✓ Optimization completed successfully!")
        logger.info(f"Optimized model saved to: {output_path}")
    else:
        logger.error("Optimization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
