#!/usr/bin/env python3
"""
Build all platform-specific packages for ls-ml-toolkit
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and check for errors"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def build_package(pyproject_file, output_dir):
    """Build a package using specific pyproject.toml"""
    print(f"\n🔨 Building package with {pyproject_file}")
    
    # Backup original pyproject.toml
    if Path("pyproject.toml").exists():
        shutil.copy2("pyproject.toml", "pyproject.toml.backup")
    
    # Copy platform-specific pyproject.toml
    shutil.copy2(pyproject_file, "pyproject.toml")
    
    try:
        # Clean previous builds
        run_command(["python", "-m", "build", "--clean"])
        
        # Move built packages to output directory
        if Path("dist").exists():
            output_dir.mkdir(exist_ok=True)
            for file in Path("dist").glob("*.whl"):
                shutil.move(str(file), str(output_dir / file.name))
            for file in Path("dist").glob("*.tar.gz"):
                shutil.move(str(file), str(output_dir / file.name))
            print(f"✅ Built packages moved to {output_dir}")
        
    finally:
        # Restore original pyproject.toml
        if Path("pyproject.toml.backup").exists():
            shutil.move("pyproject.toml.backup", "pyproject.toml")

def main():
    """Build all platform-specific packages"""
    print("🚀 Building all platform-specific packages for ls-ml-toolkit")
    print("=" * 60)
    
    # Create output directories
    base_dir = Path("dist-packages")
    base_dir.mkdir(exist_ok=True)
    
    # Build base package (CPU only)
    print("\n📦 Building base package (CPU only)")
    build_package("pyproject.toml", base_dir / "base")
    
    # Build macOS package
    print("\n🍎 Building macOS package (MPS support)")
    build_package("pyproject-macos.toml", base_dir / "macos")
    
    # Build Linux NVIDIA package
    print("\n🐧 Building Linux NVIDIA package (CUDA support)")
    build_package("pyproject-linux-nvidia.toml", base_dir / "linux-nvidia")
    
    # Build Linux AMD package
    print("\n🐧 Building Linux AMD package (ROCm support)")
    build_package("pyproject-linux-amd.toml", base_dir / "linux-amd")
    
    print("\n✅ All packages built successfully!")
    print(f"📁 Packages are in: {base_dir}")
    
    # Show package sizes
    print("\n📊 Package sizes:")
    for platform_dir in base_dir.iterdir():
        if platform_dir.is_dir():
            total_size = sum(f.stat().st_size for f in platform_dir.rglob('*') if f.is_file())
            print(f"  {platform_dir.name}: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
